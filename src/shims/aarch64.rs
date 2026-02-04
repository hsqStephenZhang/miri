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
            "neon.uminv.i8.v16i8" => {
                cal_uminv_and_write(this, abi, link_name, args, dest, 1)?;
            }
            "neon.uminv.i16.v8i16" => {
                cal_uminv_and_write(this, abi, link_name, args, dest, 2)?;
            }
            "neon.uminv.i32.v4i32" => {
                cal_uminv_and_write(this, abi, link_name, args, dest, 4)?;
            }

            // Polynomial carryless multiply 64x64 -> 128-bit result, returned as v2i64 (lo, hi).
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
                        let t =
                            this.read_immediate(&this.project_index(&table, u64::from(idx_u))?)?;
                        t.to_scalar()
                    } else {
                        Scalar::from_u8(0)
                    };
                    this.write_scalar(val, &this.project_index(&dest, i)?)?;
                }
            }

            // AES: MixColumns only (AESMC). Operates on each 128-bit chunk.
            // https://developer.arm.com/architectures/instruction-sets/intrinsics/vaesmcq_u8
            "crypto.aesmc" => {
                this.expect_target_feature_for_intrinsic(link_name, "aes")?;
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                aes_single_op(this, op, dest, |state| {
                    let mut state = aes::Block::from(state.to_le_bytes());
                    aes::hazmat::mix_columns(&mut state);
                    u128::from_le_bytes(state.into())
                })?;
            }

            // AES: InvMixColumns only (AESIMC). Operates on each 128-bit chunk.
            // https://developer.arm.com/architectures/instruction-sets/intrinsics/vaesimcq_u8
            "crypto.aesimc" => {
                this.expect_target_feature_for_intrinsic(link_name, "aes")?;
                let [op] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                aes_single_op(this, op, dest, |state| {
                    let mut state = aes::Block::from(state.to_le_bytes());
                    aes::hazmat::inv_mix_columns(&mut state);
                    u128::from_le_bytes(state.into())
                })?;
            }

            // AES: AESE (SubBytes + ShiftRows + AddRoundKey, without MixColumns).
            // Combined with AESMC equals a full encryption round.
            // https://developer.arm.com/architectures/instruction-sets/intrinsics/vaeseq_u8
            "crypto.aese" => {
                this.expect_target_feature_for_intrinsic(link_name, "aes")?;
                let [state, key] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                // AESE is the encryption last round (no MixColumns).
                aes_round(this, state, key, dest, |state, key| {
                    let mut state = aes::Block::from(state.to_le_bytes());
                    // Do a full cipher round with zero key to skip XOR,
                    // then undo MixColumns and finally XOR the actual key.
                    aes::hazmat::cipher_round(&mut state, &aes::Block::from([0; 16]));
                    aes::hazmat::inv_mix_columns(&mut state);
                    u128::from_le_bytes(state.into()) ^ key
                })?;
            }

            // AES: AESD (InvSubBytes + InvShiftRows + AddRoundKey, without InvMixColumns).
            // Combined with AESIMC equals a full decryption round.
            // https://developer.arm.com/architectures/instruction-sets/intrinsics/vaesdq_u8
            "crypto.aesd" => {
                this.expect_target_feature_for_intrinsic(link_name, "aes")?;
                let [state, key] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                // AESD is the decryption last round (no InvMixColumns).
                aes_round(this, state, key, dest, |state, key| {
                    let mut state = aes::Block::from(state.to_le_bytes());
                    // Do a full inverse cipher round with zero key to skip XOR,
                    // then undo InvMixColumns with MixColumns and finally XOR the actual key.
                    aes::hazmat::equiv_inv_cipher_round(&mut state, &aes::Block::from([0; 16]));
                    aes::hazmat::mix_columns(&mut state);
                    u128::from_le_bytes(state.into()) ^ key
                })?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}

// Unsigned minimum across all lanes -> scalar result.
fn cal_uminv_and_write<'tcx>(
    ecx: &mut crate::MiriInterpCx<'tcx>,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    link_name: Symbol,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
    size: u64,
) -> InterpResult<'tcx, ()> {
    let [op] = ecx.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
    let (op, op_len) = ecx.project_to_simd(op)?;
    assert_eq!(op_len, size);
    let size = Size::from_bytes(size);
    let mut min_val: u128 = u128::MAX;
    for i in 0..op_len {
        let v = ecx.read_immediate(&ecx.project_index(&op, i)?)?;
        let u = v.to_scalar().to_uint(size)?;
        if u < min_val {
            min_val = u;
        }
    }
    ecx.write_scalar(Scalar::from_uint(min_val, size), dest)?;
    interp_ok(())
}

// Performs an AES operation (given by `f`) on each 128-bit word of `op`.
fn aes_single_op<'tcx>(
    ecx: &mut crate::MiriInterpCx<'tcx>,
    op: &OpTy<'tcx>,
    dest: &MPlaceTy<'tcx>,
    f: impl Fn(u128) -> u128,
) -> InterpResult<'tcx, ()> {
    assert_eq!(dest.layout.size, op.layout.size);

    // Transmute arguments to arrays of `u128`.
    assert_eq!(dest.layout.size.bytes() % 16, 0);
    let len = dest.layout.size.bytes() / 16;

    let u128_array_layout = ecx.layout_of(Ty::new_array(ecx.tcx.tcx, ecx.tcx.types.u128, len))?;

    let op = op.transmute(u128_array_layout, ecx)?;
    let dest = dest.transmute(u128_array_layout, ecx)?;

    for i in 0..len {
        let state = ecx.read_scalar(&ecx.project_index(&op, i)?)?.to_u128()?;
        let dest = ecx.project_index(&dest, i)?;

        let res = f(state);

        ecx.write_scalar(Scalar::from_u128(res), &dest)?;
    }

    interp_ok(())
}

// Performs an AES round (given by `f`) on each 128-bit word of
// `state` with the corresponding 128-bit key of `key`.
fn aes_round<'tcx>(
    ecx: &mut crate::MiriInterpCx<'tcx>,
    state: &OpTy<'tcx>,
    key: &OpTy<'tcx>,
    dest: &MPlaceTy<'tcx>,
    f: impl Fn(u128, u128) -> u128,
) -> InterpResult<'tcx, ()> {
    assert_eq!(dest.layout.size, state.layout.size);
    assert_eq!(dest.layout.size, key.layout.size);

    // Transmute arguments to arrays of `u128`.
    assert_eq!(dest.layout.size.bytes() % 16, 0);
    let len = dest.layout.size.bytes() / 16;

    let u128_array_layout = ecx.layout_of(Ty::new_array(ecx.tcx.tcx, ecx.tcx.types.u128, len))?;

    let state = state.transmute(u128_array_layout, ecx)?;
    let key = key.transmute(u128_array_layout, ecx)?;
    let dest = dest.transmute(u128_array_layout, ecx)?;

    for i in 0..len {
        let state = ecx.read_scalar(&ecx.project_index(&state, i)?)?.to_u128()?;
        let key = ecx.read_scalar(&ecx.project_index(&key, i)?)?.to_u128()?;
        let dest = ecx.project_index(&dest, i)?;

        let res = f(state, key);

        ecx.write_scalar(Scalar::from_u128(res), &dest)?;
    }

    interp_ok(())
}
