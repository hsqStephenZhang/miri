use rustc_abi::{CanonAbi, Size};
use rustc_middle::mir::BinOp;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_aarch64_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.aarch64.").unwrap();
        match unprefixed_name {
            // Used to implement the vpmaxq_u8 function.
            // Computes the maximum of adjacent pairs; the first half of the output is produced from the
            // `left` input, the second half of the output from the `right` input.
            // https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxq_u8
            "neon.umaxp.v16i8" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, lane_count) = this.project_to_simd(dest)?;
                assert_eq!(left_len, right_len);
                assert_eq!(lane_count, left_len);

                for lane_idx in 0..lane_count {
                    let src = if lane_idx < (lane_count / 2) { &left } else { &right };
                    let src_idx = lane_idx.strict_rem(lane_count / 2);

                    let lhs_lane =
                        this.read_immediate(&this.project_index(src, src_idx.strict_mul(2))?)?;
                    let rhs_lane = this.read_immediate(
                        &this.project_index(src, src_idx.strict_mul(2).strict_add(1))?,
                    )?;

                    // Compute `if lhs > rhs { lhs } else { rhs }`, i.e., `max`.
                    let res_lane = if this
                        .binary_op(BinOp::Gt, &lhs_lane, &rhs_lane)?
                        .to_scalar()
                        .to_bool()?
                    {
                        lhs_lane
                    } else {
                        rhs_lane
                    };

                    let dest = this.project_index(&dest, lane_idx)?;
                    this.write_immediate(*res_lane, &dest)?;
                }
            }

            // Vector table lookup: each index selects a byte from the 16-byte table, out-of-range -> 0.
            // Semantics correspond to AArch64 TBL (single table) instruction.
            // Name as reported by LLVM: llvm.aarch64.neon.tbl1.v16i8
            "neon.tbl1.v16i8" => {
                let [table, indices] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let (table, table_len) = this.project_to_simd(table)?;
                let (indices, idx_len) = this.project_to_simd(indices)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;
                assert_eq!(table_len, 16);
                assert_eq!(idx_len, dest_len);

                let elem_size = Size::from_bytes(1);
                for i in 0..dest_len {
                    let idx = this.read_immediate(&this.project_index(&indices, i)?)?;
                    let idx_u = idx.to_scalar().to_uint(elem_size)? as u8;
                    let val = if (idx_u as usize) < table_len as usize {
                        let t = this.read_immediate(&this.project_index(&table, u64::from(idx_u))?)?;
                        t.to_scalar()
                    } else {
                        Scalar::from_u8(0)
                    };
                    this.write_scalar(val, &this.project_index(&dest, i)?)?;
                }
            }

            // Unsigned minimum across all lanes -> scalar result.
            // llvm.aarch64.neon.uminv.i8.v16i8
            "neon.uminv.i8.v16i8" => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                assert_eq!(op_len, 16);
                let size = Size::from_bytes(1);
                let mut min_val: u128 = u128::MAX;
                for i in 0..op_len {
                    let v = this.read_immediate(&this.project_index(&op, i)?)?;
                    let u = v.to_scalar().to_uint(size)?;
                    if u < min_val { min_val = u; }
                }
                this.write_scalar(Scalar::from_uint(min_val, size), dest)?;
            }
            // llvm.aarch64.neon.uminv.i16.v8i16
            "neon.uminv.i16.v8i16" => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                assert_eq!(op_len, 8);
                let size = Size::from_bytes(2);
                let mut min_val: u128 = u128::MAX;
                for i in 0..op_len {
                    let v = this.read_immediate(&this.project_index(&op, i)?)?;
                    let u = v.to_scalar().to_uint(size)?;
                    if u < min_val { min_val = u; }
                }
                this.write_scalar(Scalar::from_uint(min_val, size), dest)?;
            }
            // llvm.aarch64.neon.uminv.i32.v4i32
            "neon.uminv.i32.v4i32" => {
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let (op, op_len) = this.project_to_simd(op)?;
                assert_eq!(op_len, 4);
                let size = Size::from_bytes(4);
                let mut min_val: u128 = u128::MAX;
                for i in 0..op_len {
                    let v = this.read_immediate(&this.project_index(&op, i)?)?;
                    let u = v.to_scalar().to_uint(size)?;
                    if u < min_val { min_val = u; }
                }
                this.write_scalar(Scalar::from_uint(min_val, size), dest)?;
            }

            // Polynomial carryless multiply 64x64 -> 128-bit result, returned as v2i64 (lo, hi).
            // llvm.aarch64.neon.pmull64
            "neon.pmull64" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let (left, left_len) = this.project_to_simd(left)?;
                let (right, right_len) = this.project_to_simd(right)?;
                let (dest, dest_len) = this.project_to_simd(dest)?;

                // Expect v1i64 x v1i64 -> v2i64
                assert_eq!(left_len, 1);
                assert_eq!(right_len, 1);
                assert_eq!(dest_len, 2);

                let a = this.read_immediate(&this.project_index(&left, 0)?)?;
                let b = this.read_immediate(&this.project_index(&right, 0)?)?;
                let a_u = a.to_scalar().to_uint(Size::from_bytes(8))? as u64;
                let b_u = b.to_scalar().to_uint(Size::from_bytes(8))? as u64;

                let a128 = a_u as u128;
                let mut res: u128 = 0;
                let mut bb = b_u;
                let mut shift = 0u32;
                while bb != 0 {
                    if (bb & 1) != 0 {
                        res ^= a128 << shift;
                    }
                    bb >>= 1;
                    shift += 1;
                }

                let lo = (res & 0xffff_ffff_ffff_ffffu128) as u64;
                let hi = (res >> 64) as u64;
                let dest0 = this.project_index(&dest, 0)?;
                let dest1 = this.project_index(&dest, 1)?;
                this.write_scalar(Scalar::from_uint(lo, Size::from_bytes(8)), &dest0)?;
                this.write_scalar(Scalar::from_uint(hi, Size::from_bytes(8)), &dest1)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
