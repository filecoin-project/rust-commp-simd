# sha256_reg_aliases.h - symbolic register names for SHA256 x16 AVX-512
#
# Matches the register assignment from intel-ipsec-mb / minio Plan9.

#define SHA256_DIGEST_ROW_SIZE 64

# ---- SHA256 state (Structure-of-Arrays, 16 lanes each) ----
#define A       %zmm0
#define B       %zmm1
#define C       %zmm2
#define D       %zmm3
#define E       %zmm4
#define F       %zmm5
#define G       %zmm6
#define H       %zmm7

# ---- Scratch registers ----
#define T1      %zmm8
#define TMP0    %zmm9
#define TMP1    %zmm10
#define TMP2    %zmm11
#define TMP3    %zmm12
#define TMP4    %zmm13
#define TMP5    %zmm14
#define TMP6    %zmm15

# ---- Message schedule W0-W15 (zmm) ----
#define W0      %zmm16
#define W1      %zmm17
#define W2      %zmm18
#define W3      %zmm19
#define W4      %zmm20
#define W5      %zmm21
#define W6      %zmm22
#define W7      %zmm23
#define W8      %zmm24
#define W9      %zmm25
#define W10     %zmm26
#define W11     %zmm27
#define W12     %zmm28
#define W13     %zmm29
#define W14     %zmm30
#define W15     %zmm31

# ---- Message schedule W0-W15 (ymm, for split-half loads) ----
#define YW0     %ymm16
#define YW1     %ymm17
#define YW2     %ymm18
#define YW3     %ymm19
#define YW4     %ymm20
#define YW5     %ymm21
#define YW6     %ymm22
#define YW7     %ymm23
#define YW8     %ymm24
#define YW9     %ymm25
#define YW10    %ymm26
#define YW11    %ymm27
#define YW12    %ymm28
#define YW13    %ymm29
#define YW14    %ymm30
#define YW15    %ymm31

# ---- GPRz ----
#define STATE    %rdi
#define INP_SIZE %rsi
#define IDX      %rcx
#define TBL      %rdx
