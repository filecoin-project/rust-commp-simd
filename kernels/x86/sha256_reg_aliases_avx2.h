# sha256_reg_aliases_avx2.h - symbolic register names for SHA256 x8 AVX2
#
# Matches intel-ipsec-mb sha256_oct_avx2 register assignment
# 16 YMM registers total - all of em' used

# ---- 8 lanes x 4 bytes = 32 bytes = one YMM width (256 bits)
#define SHA256_DIGEST_ROW_SIZE 32
#define X8_SLOT 32      // size of one W entry on stack (1 x YMM = 8 lanes x 4 bytes)

# ---- SHA256 state (SoA, 8 lanes eaach) ----
#define  A       %ymm0
#define  B       %ymm1
#define  C       %ymm2
#define  D       %ymm3
#define  E       %ymm4
#define  F       %ymm5
#define  G       %ymm6
#define  H       %ymm7
// (8 / 16) registers used

# ---- Scratch registers ----
#define T1       %ymm8
#define TMP0     %ymm9
#define TMP1     %ymm10
#define TMP2     %ymm11
#define TMP3     %ymm12
#define TMP4     %ymm13
#define TMP5     %ymm14
#define TMP6     %ymm15
// (16 / 16) registers used

/* W0-W15 will live on the stack, not in registers.
   AVX2 has only 16 YMMs - all used by state + scratch.
   Stack slots: 16 x 32 bytes = 512 bytes at rsp+0..rsp+0x1FF

   Below are assumptions made behind this register choice:

   Rationale #1:
      State registers A-H are read and written every single round.
      64 rounds. A round touches all 8 of them plus needs scratch for the SIGMA/CH/MAJ
      That's ~36 instructions per round all hitting state and scratch
      If state lived on stack every round would be
      vmovdqu loads and stores for 8 values in and out

   Rationale #2:
      W values (W0-W15) are accessed less. Each W is

      written once at:
         - round 0-15
         - messagee schedule

      read once (during it's round)
      +
      read x4 times (Wt-2, Wt-7, Wt-15, Wt-16 for the message schedule)
      (in formula: Wt = sigma1(Wt-2) + Wt-7 + sigma0(Wt-15) + Wt-16)

      vmovdqa to/from stack is cheap
      (512 bytes probably stays resident in L1 cache for all 64 rounds)
       512 = 16 W slots x 32 bytes per slot

      State and scratch are hot (read+written every round, all 64 rounds)
      W on stack is warm (written once, read a few times by later rounds)
*/

# ---- GPRz ----
#define STATE    %rdi
#define INP_SIZE %rsi
#define IDX      %rcx
#define TBL      %rdx
