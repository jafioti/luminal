// Copyright © 2023 Apple Inc.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)    \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }                                                                       \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                 \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);         \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(            \
      addr_space _MLX_BFloat16& lhs, itype rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }                                                                       \
  constexpr METAL_FUNC addr_space itype& __operator__(                    \
      addr_space itype& lhs, _MLX_BFloat16 rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype) \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                             \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype); \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype); \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype); \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(     \
      addr_space _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) {          \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);  \
    return lhs;                                                    \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__) \
  bfloat_inplace_op_helper(__op__, __operator__, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

/////////////////////////////////////////////////////////////////////////////
// Bfloat numeric limits
/////////////////////////////////////////////////////////////////////////////

#pragma METAL internals : enable

namespace metal {

template <>
struct _numeric_limits_impl<bfloat16_t> : _fp_numeric_limits_impl_base {
  static constexpr constant int digits = 8;
  static constexpr constant int digits10 = 2;
  static constexpr constant int max_digits10 = 4;
  static constexpr constant int radix = 2;
  static constexpr constant int min_exponent = -125;
  static constexpr constant int min_exponent10 = -37;
  static constexpr constant int max_exponent = 128;
  static constexpr constant int max_exponent10 = 38;

  static constexpr bfloat16_t min() {
    return _MLX_BFloat16(0x0080, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t lowest() {
    return _MLX_BFloat16(0xFF7F, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t max() {
    return _MLX_BFloat16(0x7F7F, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t epsilon() {
    return _MLX_BFloat16(0x3C00, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t round_error() {
    return _MLX_BFloat16(0x3F00, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t infinity() {
    return _MLX_BFloat16(0x7F80, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t quiet_NaN() {
    return _MLX_BFloat16(0x7FC0, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t signaling_NaN() {
    return _MLX_BFloat16(0x7F80, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t denorm_min() {
    return _MLX_BFloat16(0x0001, _MLX_BFloat16::bits_to_bfloat());
  }
};

METAL_FUNC bool isnan(_MLX_BFloat16 x) {
  return x != x;
}

} // namespace metal

#pragma METAL internals : disable

#endif // defined(__HAVE_BFLOAT__)


///////////////////////////////////////////////////////////////////////////////
// Metal math for bfloat16
///////////////////////////////////////////////////////////////////////////////

/*

Following the Metal Shading Language Specification (Metal 3.1)

"bfloat is an extended itypeing point type that only allows implicit conversion
 to a type of greater itypeing point rank. While bfloat can be implicitly
 converted to itype, it cannot be implicitly converted to half, and neither
 itype nor half can be implicitly converted to bfloat."

Further, as far as I can tell, the stdlib math/simd functions are not defined
for bfloat and calling with an argument of type bfloat will result in that
argument getting implicitly converted to itype which then returns an output
that is (likely) a itype which cannot be implicitly converted into a bfloat

This leads to situations where
bfloat a = 5.0bf;
bfloat b = metal::abs(a); // this will throw an error since abs return itype
bfloat c = static_cast<bfloat>(metal::abs(a)); // this is fine

For the moment, I will be adding overloaded instantiations of the math
functions to accordingly automatically handle the casting

*/

#define instantiate_metal_math_funcs(itype, otype, ctype, mfast)               \
                                                                               \
  METAL_FUNC otype abs(itype x) {                                              \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acos(itype x) {                                             \
    return static_cast<otype>(__metal_acos(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acosh(itype x) {                                            \
    return static_cast<otype>(__metal_acosh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype asin(itype x) {                                             \
    return static_cast<otype>(__metal_asin(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype asinh(itype x) {                                            \
    return static_cast<otype>(__metal_asinh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype atan(itype y_over_x) {                                      \
    return static_cast<otype>(                                                 \
        __metal_atan(static_cast<ctype>(y_over_x), mfast));                    \
  }                                                                            \
  METAL_FUNC otype atan2(itype y, itype x) {                                   \
    return static_cast<otype>(                                                 \
        __metal_atan2(static_cast<ctype>(y), static_cast<ctype>(x), mfast));   \
  }                                                                            \
  METAL_FUNC otype atanh(itype x) {                                            \
    return static_cast<otype>(__metal_atanh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype ceil(itype x) {                                             \
    return static_cast<otype>(__metal_ceil(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cos(itype x) {                                              \
    return static_cast<otype>(__metal_cos(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype cosh(itype x) {                                             \
    return static_cast<otype>(__metal_cosh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cospi(itype x) {                                            \
    return static_cast<otype>(__metal_cospi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype divide(itype x, itype y) {                                  \
    return static_cast<otype>(                                                 \
        __metal_divide(static_cast<ctype>(x), static_cast<ctype>(y), mfast));  \
  }                                                                            \
  METAL_FUNC otype exp(itype x) {                                              \
    return static_cast<otype>(__metal_exp(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype exp10(itype x) {                                            \
    return static_cast<otype>(__metal_exp10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype exp2(itype x) {                                             \
    return static_cast<otype>(__metal_exp2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fabs(itype x) {                                             \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fdim(itype x, itype y) {                                    \
    ctype t = static_cast<ctype>(x - y);                                       \
    return static_cast<otype>(select(t, ctype(0), t < ctype(0) || x == y));    \
  }                                                                            \
  METAL_FUNC otype floor(itype x) {                                            \
    return static_cast<otype>(__metal_floor(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype fma(itype x, itype y, itype z) {                            \
    return static_cast<otype>(__metal_fma(                                     \
        static_cast<ctype>(x), static_cast<ctype>(y), static_cast<ctype>(z))); \
  }                                                                            \
  METAL_FUNC otype fmax(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmax3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmedian3(itype x, itype y, itype z) {                       \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmin(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmin3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmod(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmod(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fract(itype x) {                                            \
    return static_cast<otype>(__metal_fract(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype frexp(itype x, thread int& exp) {                           \
    return static_cast<otype>(__metal_frexp(static_cast<ctype>(x), &exp));     \
  }                                                                            \
  METAL_FUNC otype ldexp(itype x, int k) {                                     \
    return static_cast<otype>(__metal_ldexp(static_cast<ctype>(x), k, mfast)); \
  }                                                                            \
  METAL_FUNC otype log(itype x) {                                              \
    return static_cast<otype>(__metal_log(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype log10(itype x) {                                            \
    return static_cast<otype>(__metal_log10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype log2(itype x) {                                             \
    return static_cast<otype>(__metal_log2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype max(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype max3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype median3(itype x, itype y, itype z) {                        \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype min(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype min3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype nextafter(itype x, itype y) {                               \
    return static_cast<otype>(                                                 \
        __metal_nextafter(static_cast<ctype>(x), static_cast<ctype>(y)));      \
  }                                                                            \
  METAL_FUNC otype pow(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_pow(static_cast<ctype>(x), static_cast<ctype>(y), mfast));     \
  }                                                                            \
  METAL_FUNC otype powr(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_powr(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype rint(itype x) {                                             \
    return static_cast<otype>(__metal_rint(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype round(itype x) {                                            \
    return static_cast<otype>(__metal_round(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype rsqrt(itype x) {                                            \
    return static_cast<otype>(__metal_rsqrt(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sin(itype x) {                                              \
    return static_cast<otype>(__metal_sin(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype sinh(itype x) {                                             \
    return static_cast<otype>(__metal_sinh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype sinpi(itype x) {                                            \
    return static_cast<otype>(__metal_sinpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sqrt(itype x) {                                             \
    return static_cast<otype>(__metal_sqrt(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tan(itype x) {                                              \
    return static_cast<otype>(__metal_tan(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype tanh(itype x) {                                             \
    return static_cast<otype>(__metal_tanh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tanpi(itype x) {                                            \
    return static_cast<otype>(__metal_tanpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype trunc(itype x) {                                            \
    return static_cast<otype>(__metal_trunc(static_cast<ctype>(x), mfast));    \
  }

namespace metal {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_MAYBE_FAST_MATH__);

namespace fast {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_FAST_MATH__);

} // namespace fast

namespace precise {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_PRECISE_MATH__);

} // namespace precise

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Metal simd for bfloat16
///////////////////////////////////////////////////////////////////////////////

#define instantiate_metal_simd_comm_funcs(                                   \
    itype, otype, ctype, itype_to_ctype, ctype_to_otype)                     \
                                                                             \
  METAL_FUNC otype simd_broadcast(itype data, ushort broadcast_lane_id) {    \
    return ctype_to_otype(                                                   \
        __metal_simd_broadcast(itype_to_ctype(data), broadcast_lane_id));    \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle(itype data, ushort simd_lane_id) {           \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle(itype_to_ctype(data), simd_lane_id));           \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_down(itype data, ushort delta) {             \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_down(itype_to_ctype(data), delta));             \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_down(itype data, ushort delta) {      \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_down(itype_to_ctype(data), delta));      \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_up(itype data, ushort delta) {        \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_up(itype_to_ctype(data), delta));        \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_up(itype data, ushort delta) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_up(itype_to_ctype(data), delta));               \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_xor(itype data, ushort mask) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_xor(itype_to_ctype(data), mask));               \
  }

#define instantiate_metal_simd_reduction_funcs(itype, otype, ctype)            \
                                                                               \
  METAL_FUNC otype simd_max(itype data) {                                      \
    return static_cast<otype>(__metal_simd_max(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_min(itype data) {                                      \
    return static_cast<otype>(__metal_simd_min(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_product(itype data) {                                  \
    return static_cast<otype>(__metal_simd_product(static_cast<ctype>(data))); \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_sum(itype data) {                                      \
    return static_cast<otype>(__metal_simd_sum(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_xor(itype data) {                                      \
    return static_cast<otype>(__metal_simd_xor(static_cast<ctype>(data)));     \
  }

#if defined(__HAVE_BFLOAT__)

#define bfloat16_to_uint16(x) as_type<uint16_t>(x)
#define uint16_to_bfloat16(x) as_type<bfloat16_t>(x)

#else

#define bfloat16_to_uint16(x) x.bits_
#define uint16_to_bfloat16(x) _MLX_BFloat16(x, _MLX_BFloat16::bits_to_bfloat())

#endif

namespace metal {

instantiate_metal_simd_comm_funcs(
    bfloat16_t,
    bfloat16_t,
    uint16_t,
    bfloat16_to_uint16,
    uint16_to_bfloat16);
instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);

} // namespace metal

#ifdef __METAL__
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_BINARY_SPECIALIZED_DIMS = 5;
static MTL_CONST constexpr int MAX_COPY_SPECIALIZED_DIMS = 5;
static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 16;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int SOFTMAX_LOOPED_LIMIT = 4096;


using namespace metal;

struct complex64_t;

template <typename T>
static constexpr constant bool can_convert_to_complex64 =
    !is_same_v<T, complex64_t> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_complex64 =
    !is_same_v<T, complex64_t> &&
    (is_convertible_v<float, T> || is_convertible_v<bfloat16_t, T>);

struct complex64_t {
  float real;
  float imag;

  // Constructors
  constexpr complex64_t(float real, float imag) : real(real), imag(imag){};

  // Conversions to complex64_t
  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) thread : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) threadgroup : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) device : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) constant : real(x), imag(0) {}

  // Conversions from complex64_t
  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const thread {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const threadgroup {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const device {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const constant {
    return static_cast<T>(real);
  }
};

constexpr complex64_t operator-(complex64_t x) {
  return {-x.real, -x.imag};
}

constexpr bool operator>=(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag >= b.imag);
}

constexpr bool operator>(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag > b.imag);
}

constexpr bool operator<=(complex64_t a, complex64_t b) {
  return operator>=(b, a);
}

constexpr bool operator<(complex64_t a, complex64_t b) {
  return operator>(b, a);
}

constexpr bool operator==(complex64_t a, complex64_t b) {
  return a.real == b.real && a.imag == b.imag;
}

constexpr complex64_t operator+(complex64_t a, complex64_t b) {
  return {a.real + b.real, a.imag + b.imag};
}

constexpr complex64_t operator-(complex64_t a, complex64_t b) {
  return {a.real - b.real, a.imag - b.imag};
}

constexpr complex64_t operator*(complex64_t a, complex64_t b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

constexpr complex64_t operator/(complex64_t a, complex64_t b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a.real * b.real + a.imag * b.imag;
  auto y = a.imag * b.real - a.real * b.imag;
  return {x / denom, y / denom};
}

constexpr complex64_t operator%(complex64_t a, complex64_t b) {
  auto real = a.real - (b.real * static_cast<int64_t>(a.real / b.real));
  auto imag = a.imag - (b.imag * static_cast<int64_t>(a.imag / b.imag));
  return {real, imag};
}

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static const constant U max;
  static const constant U min;
  static const constant U finite_max;
  static const constant U finite_min;
};

#define instantiate_default_limit(type)                                      \
  template <>                                                                \
  struct Limits<type> {                                                      \
    static constexpr constant type max = metal::numeric_limits<type>::max(); \
    static constexpr constant type min = metal::numeric_limits<type>::min(); \
    static constexpr constant type finite_max =                              \
        metal::numeric_limits<type>::max();                                  \
    static constexpr constant type finite_min =                              \
        metal::numeric_limits<type>::min();                                  \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)             \
  template <>                                     \
  struct Limits<type> {                           \
    static constexpr constant type max =          \
        metal::numeric_limits<type>::infinity();  \
    static constexpr constant type min =          \
        -metal::numeric_limits<type>::infinity(); \
    static constexpr constant type finite_max =   \
        metal::numeric_limits<type>::max();       \
    static constexpr constant type finite_min =   \
        -metal::numeric_limits<type>::max();      \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);

template <>
struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

inline size_t elem_to_loc(
    uint elem,
    device const int* shape,
    device const size_t* strides,
    int ndim) {
  size_t loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

inline size_t elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const size_t* strides,
    int ndim) {
  size_t loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <int NDIM>
inline uint2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const size_t a_strides[NDIM],
    constant const size_t b_strides[NDIM]) {
  uint2 loc = {
      static_cast<uint>(
          elem.x * a_strides[NDIM - 1] + elem.y * a_strides[NDIM - 2]),
      static_cast<uint>(
          elem.x * b_strides[NDIM - 1] + elem.y * b_strides[NDIM - 2])};
  for (int d = NDIM - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

template <int NDIM>
inline size_t elem_to_loc_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const size_t strides[NDIM]) {
  size_t loc = elem.x * strides[NDIM - 1] + elem.y * strides[NDIM - 2];
  for (int d = NDIM - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

inline size_t elem_to_loc_1(uint elem, constant const size_t& stride) {
  return elem * stride;
}

inline size_t elem_to_loc_2(uint2 elem, constant const size_t strides[2]) {
  return elem.x * strides[1] + elem.y * strides[0];
}

inline size_t elem_to_loc_3(uint3 elem, constant const size_t strides[3]) {
  return elem.x * strides[2] + elem.y * strides[1] + elem.z * strides[0];
}

// Non templated version to handle arbitrary dims
inline size_t elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const size_t* strides,
    int ndim) {
  size_t loc = elem.x * strides[ndim - 1] + elem.y * strides[ndim - 2];
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

inline uint2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    int ndim) {
  uint2 loc = {
      static_cast<uint>(
          elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2]),
      static_cast<uint>(
          elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2])};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

template <int NDIM>
inline uint elem_to_loc_nd(
    uint elem,
    device const int* shape,
    device const size_t* strides);

template <>
inline uint elem_to_loc_nd<1>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  return (elem % shape[0]) * strides[0];
}

template <>
inline uint elem_to_loc_nd<2>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  uint loc = (elem % shape[1]) * strides[1];
  elem /= shape[1];
  loc += (elem % shape[0]) * strides[0];
  return loc;
}

template <>
inline uint elem_to_loc_nd<3>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  uint loc = (elem % shape[2]) * strides[2];
  elem /= shape[2];
  loc += (elem % shape[1]) * strides[1];
  elem /= shape[1];
  loc += (elem % shape[0]) * strides[0];
  return loc;
}

template <>
inline uint elem_to_loc_nd<4>(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  uint loc = (elem % shape[3]) * strides[3];
  elem /= shape[3];
  loc += (elem % shape[2]) * strides[2];
  elem /= shape[2];
  loc += (elem % shape[1]) * strides[1];
  elem /= shape[1];
  loc += (elem % shape[0]) * strides[0];
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
inline size_t ceildiv(size_t N, size_t M) {
  return (N + M - 1) / M;
}

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  float xp1 = 1.0f + x;
  return (xp1 == 1.0f) ? x : x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline bfloat16_t log1p(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  bfloat16_t ret =
      (xp1 == 1.0f) ? x : bfloat16_t(x * (metal::log(xp1) / (xp1 - 1.0f)));
  return ret;
}

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

template <
  typename T,
  const int BM, /* Threadgroup rows (in threads) */
  const int BN, /* Threadgroup cols (in threads) */
  const int TM, /* Thread rows (in elements) */
  const int TN > /* Thread cols (in elements) */
struct GEMVKernel {

  static_assert(BN == SIMD_SIZE, "gemv block must have a width of SIMD_SIZE");

  // - The matrix of size (M = out_vec_size, N = in_vec_size) is divided up
  //   into blocks of (BM * TM, BN * TN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each thead group is launched with (BN, BM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows
  //      and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across the rows
  //      These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid will have blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted inwards
  //     such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN * TN * 2;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      device T* out_vec,
      const constant int& in_vec_size [[buffer(3)]],
      const constant int& out_vec_size [[buffer(4)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {

    // Appease compiler
    (void)lid;

    // Threadgroup in_vec cache
    threadgroup T* in_vec_block = tgp_memory + simd_lid * TN * 2;

    // Thread local accumulation results
    thread T result[TM] = {0};
    thread T inter[TN];
    thread T v_coeff[TN];

    // Block position
    int out_row = (tid.x * BM + simd_gid) * TM;

    // Exit simdgroup if rows out of bound
    if(out_row >= out_vec_size)
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Advance matrix
    mat += out_row * in_vec_size;

    // Loop over in_vec in blocks of BN * TN
    for(int bn = simd_lid * TN; bn < in_vec_size; bn += BN * TN) {

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Prefetch in_vector for threadgroup use
      if(simd_gid == 0) {
        // Main load loop
        if(bn + TN <= in_vec_size) {

          #pragma clang loop unroll(full)
          for(int tn = 0; tn < TN; tn++) {
            in_vec_block[tn] = in_vec[bn + tn];
          }

        } else { // Edgecase

          #pragma clang loop unroll(full)
          for(int tn = 0; tn < TN; tn++) {
            in_vec_block[tn] = bn + tn < in_vec_size ? in_vec[bn + tn] : 0;
          }

        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Load for all rows
      #pragma clang loop unroll(full)
      for(int tn = 0; tn < TN; tn++) {
        v_coeff[tn] = in_vec_block[tn];
      }

      // Per thread work loop
      #pragma clang loop unroll(full)
      for(int tm = 0; tm < TM; tm++) {

        // Load for the row
        for(int tn = 0; tn < TN; tn++) {
          inter[tn] = mat[tm * in_vec_size + bn + tn];
        }

        // Accumulate results
        for(int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }

      }
    }

    // Simdgroup accumulations
    #pragma clang loop unroll(full)
    for(int tm = 0; tm < TM; tm++) {
      result[tm] = simd_sum(result[tm]);
    }

    // Write outputs
    if(simd_lid == 0) {

      #pragma clang loop unroll(full)
      for(int tm = 0; tm < TM; tm++) {
        out_vec[out_row + tm] = result[tm];
      }

    }

  }

};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  const int BM, /* Threadgroup rows (in threads) */
  const int BN, /* Threadgroup cols (in threads) */
  const int TM, /* Thread rows (in elements) */
  const int TN > /* Thread cols (in elements) */
struct GEMVTKernel {

  // - The matrix of size (M = in_vec_size, N = out_vec_size) is divided up
  //   into blocks of (BM * TM, BN * TN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each thead group is launched with (BN, BM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows
  //      and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across the rows
  //      These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid will have blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted inwards
  //     such that the thread block fits exactly in the matrix


  MLX_MTL_CONST short tgp_mem_size = BN * BM * TN;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      device T* out_vec,
      const constant int& in_vec_size [[buffer(3)]],
      const constant int& out_vec_size [[buffer(4)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {

    // Appease compiler
    (void)simd_gid;
    (void)simd_lid;

    // Thread local accumulation results
    T result[TN] = {0};
    T inter[TN];
    T v_coeff[TM];

    // Threadgroup accumulation results
    threadgroup T* tgp_results = tgp_memory + lid.x * BM * TN;

    int out_col = (tid.x * BN + lid.x) * TN;
    int in_row = lid.y * TM;

    // Edgecase handling
    if (out_col < out_vec_size) {

      out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;

      // Per thread accumulation main loop
      int bm = in_row;
      for(; bm < in_vec_size; bm += BM * TM) {
        // Adding a threadgroup_barrier improves performance slightly
        // This is possibly it may help exploit cache better
        threadgroup_barrier(mem_flags::mem_none);

        if(bm + TM <= in_vec_size) {

          #pragma clang loop unroll(full)
          for(int tm = 0; tm < TM; tm++) {
            v_coeff[tm] = in_vec[bm + tm];
          }

          #pragma clang loop unroll(full)
          for(int tm = 0; tm < TM; tm++) {
            for(int tn = 0; tn < TN; tn++) {
              inter[tn] = mat[(bm + tm) * out_vec_size + out_col + tn];
            }
            for(int tn = 0; tn < TN; tn++) {
              result[tn] += v_coeff[tm] * inter[tn];
            }
          }

        } else { // Edgecase handling
          for(int tm = 0; bm + tm < in_vec_size; tm++) {
            v_coeff[tm] = in_vec[bm + tm];

            for(int tn = 0; tn < TN; tn++) {
              inter[tn] = mat[(bm + tm) * out_vec_size + out_col + tn];
            }
            for(int tn = 0; tn < TN; tn++) {
              result[tn] += v_coeff[tm] * inter[tn];
            }

          }
        }
      }

    }

    // Threadgroup collection

    #pragma clang loop unroll(full)
    for(int i = 0; i < TN; i++) {
      tgp_results[lid.y * TN + i] = result[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup accumulation and writing out results
    if(lid.y == 0 && out_col < out_vec_size) {

      #pragma clang loop unroll(full)
      for(int i = 1; i < BM; i++) {

        #pragma clang loop unroll(full)
        for(int j = 0; j < TN; j++) {
          result[j] += tgp_results[i * TN + j];
        }
      }

      #pragma clang loop unroll(full)
      for(int j = 0; j < TN; j++) {
        out_vec[out_col + j] = result[j];
      }
    }

  }


};

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& vector_batch_stride [[buffer(5)]],
    const constant int& matrix_batch_stride [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += tid.z * vector_batch_stride;
  mat += tid.z * matrix_batch_stride;
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
    mat,
    in_vec,
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );

}

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv_nc(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& nc_dim [[buffer(5)]],
    const device int* nc_shape [[buffer(6)]],
    const device size_t* nc_strides_vec [[buffer(7)]],
    const device size_t* nc_strides_mat [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += elem_to_loc(tid.z, nc_shape, nc_strides_vec, nc_dim);
  mat += elem_to_loc(tid.z, nc_shape, nc_strides_mat, nc_dim);
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
    mat,
    in_vec,
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );

}


#define instantiate_gemv_c(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn)]] \
  [[kernel]] void gemv<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& vector_batch_stride [[buffer(5)]], \
    const constant int& matrix_batch_stride [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_nc(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn "_nc")]] \
  [[kernel]] void gemv_nc<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& nc_dim [[buffer(5)]], \
    const device int* nc_shape [[buffer(6)]], \
    const device size_t* nc_strides_vec [[buffer(7)]], \
    const device size_t* nc_strides_mat [[buffer(8)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_c(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_nc(name, itype, bm, bn, tm, tn)

#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 4, 32, 1, 4) \
  instantiate_gemv(name, itype, 4, 32, 4, 4) \
  instantiate_gemv(name, itype, 8, 32, 4, 4)

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& vector_batch_stride [[buffer(5)]],
    const constant int& matrix_batch_stride [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVTKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += tid.z * vector_batch_stride;
  mat += tid.z * matrix_batch_stride;
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
    mat,
    in_vec,
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );
}

template <
    typename T,
    const int BM, /* Threadgroup rows (in threads) */
    const int BN, /* Threadgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN)]] void gemv_t_nc(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& nc_dim [[buffer(5)]],
    const device int* nc_shape [[buffer(6)]],
    const device size_t* nc_strides_vec [[buffer(7)]],
    const device size_t* nc_strides_mat [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  using gemv_kernel = GEMVTKernel<T, BM, BN, TM, TN>;
  threadgroup T tgp_memory[gemv_kernel::tgp_mem_size];

  // Update batch offsets
  in_vec += elem_to_loc(tid.z, nc_shape, nc_strides_vec, nc_dim);
  mat += elem_to_loc(tid.z, nc_shape, nc_strides_mat, nc_dim);
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
    mat,
    in_vec,
    out_vec,
    in_vec_size,
    out_vec_size,
    tgp_memory,
    tid,
    lid,
    simd_gid,
    simd_lid
  );

}

#define instantiate_gemv_t_c(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_t_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn)]] \
  [[kernel]] void gemv_t<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& vector_batch_stride [[buffer(5)]], \
    const constant int& matrix_batch_stride [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_t_nc(name, itype, bm, bn, tm, tn) \
  template [[host_name("gemv_t_" #name "_bm" #bm "_bn" #bn "_tm" #tm "_tn" #tn "_nc")]] \
  [[kernel]] void gemv_t_nc<itype, bm, bn, tm, tn>( \
    const device itype* mat [[buffer(0)]], \
    const device itype* vec [[buffer(1)]], \
    device itype* out [[buffer(2)]], \
    const constant int& in_vec_size [[buffer(3)]], \
    const constant int& out_vec_size [[buffer(4)]], \
    const constant int& nc_dim [[buffer(5)]], \
    const device int* nc_shape [[buffer(6)]], \
    const device size_t* nc_strides_vec [[buffer(7)]], \
    const device size_t* nc_strides_mat [[buffer(8)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 lid [[thread_position_in_threadgroup]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_t(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_t_c(name, itype, bm, bn, tm, tn) \
  instantiate_gemv_t_nc(name, itype, bm, bn, tm, tn)

#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 8, 8, 4, 1) \
  instantiate_gemv_t(name, itype, 8, 8, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 16, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 32, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 64, 4, 4) \
  instantiate_gemv_t(name, itype, 8, 128, 4, 4)

instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t);
