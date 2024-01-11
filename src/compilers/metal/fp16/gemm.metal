// Copyright © 2023 Apple Inc.

///////////////////////////////////////////////////////////////////////////////
// START BF16.H
///////////////////////////////////////////////////////////////////////////////
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_stdlib>

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
// END bf16.H
///////////////////////////////////////////////////////////////////////////////

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
///////////////////////////////////////////////////////////////////////////////
// START GEMM.H
///////////////////////////////////////////////////////////////////////////////
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#define MLX_MTL_CONST static constant constexpr const

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BROWS,
    int BCOLS,
    int BK,
    int vec_size,
    int tgp_size,
    bool transpose,
    bool ldK,
    int tgp_padding = 0>
struct BlockLoader {
  // Destination dimensions
  MLX_MTL_CONST int dst_fd = transpose ? BCOLS : BROWS;
  MLX_MTL_CONST int dst_ld = (transpose ? BROWS : BCOLS) + tgp_padding;
  MLX_MTL_CONST int n_vecs = (transpose ? BROWS : BCOLS) / vec_size;

  // Stride along block row within the block
  MLX_MTL_CONST int bstride = tgp_size / n_vecs;

  // Leading dimension for src
  const int src_ld;
  // Stride along reduction axis between blocks
  const int tstride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tstride(
            BK * ((int)(transpose ^ !ldK) * src_ld + (int)(transpose ^ ldK))),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / n_vecs),
        bj(vec_size * (thread_idx % n_vecs)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
#pragma clang loop unroll(full)
    for (short i = 0; i < dst_fd; i += bstride) {
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = transpose ? src_tile_dim.yx : src_tile_dim.xy;

    // Iterate over rows of block
#pragma clang loop unroll(full)
    for (short i = 0; i < dst_fd; i += bstride) {
      // Row is in bounds, we check against column
      if ((bi + i) < src_tile_dim.y) {
        // Use fast thread memory for bound checks
        short tmp_idx[vec_size];
        T tmp_val[vec_size];

        // Make sure tmp_idx only contains valid indices
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          tmp_idx[j] = bj + j < src_tile_dim.x ? j : 0;
        }

        // Read all valid indices into tmp_val
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          tmp_val[j] = src[i * src_ld + tmp_idx[j]];
        }

        // Zero out unneeded values
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          tmp_val[j] = bj + j < src_tile_dim.x ? tmp_val[j] : T(0);
        }

        // Copy values to threadgroup memory
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = tmp_val[j];
        }
      }

      // Row is out of bounds, we just fill tgp memory with zeros
      else {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tstride;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Transforms
///////////////////////////////////////////////////////////////////////////////

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    int tgp_padding_a = 0,
    int tgp_padding_b = 0,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<T, AccumType>>
struct BlockMMA {
  // Warp tile size along M
  MLX_MTL_CONST int TM = BM / (WM * 8);
  // Warp tile size along N
  MLX_MTL_CONST int TN = BN / (WN * 8);

  // Warp tile simdgroup matrix strides along M
  MLX_MTL_CONST int TM_stride = 8 * WM;
  // Warp tile simdgroup matrix strides along M
  MLX_MTL_CONST int TN_stride = 8 * WN;

  // Leading dimensions of threadgroup A, B blocks
  MLX_MTL_CONST int lda_tgp = (transpose_a ? BM : BK) + tgp_padding_a;
  MLX_MTL_CONST int ldb_tgp = (transpose_b ? BK : BN) + tgp_padding_b;

  // Strides of A, B along reduction axis
  MLX_MTL_CONST short simd_stride_a =
      transpose_a ? TM_stride : TM_stride * lda_tgp;
  MLX_MTL_CONST short simd_stride_b =
      transpose_b ? TN_stride * ldb_tgp : TN_stride;

  // Jump between elements
  MLX_MTL_CONST short jump_a = transpose_a ? lda_tgp : 1;
  MLX_MTL_CONST short jump_b = transpose_b ? ldb_tgp : 1;

  // Offsets within threadgroup
  const int tm;
  const int tn;

  // Simdgroup matrices
  simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
  simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
  simdgroup_matrix<AccumType, 8, 8> results[TM * TN] = {
      simdgroup_matrix<AccumType, 8, 8>(0)};

  short sm;
  short sn;

  /* Constructor */
  METAL_FUNC BlockMMA(
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : tm(8 * (simd_group_id / WN)), tn(8 * (simd_group_id % WN)) {
    short qid = simd_lane_id / 4;
    sm = (qid & 4) + (simd_lane_id / 2) % 4;
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
// Iterate over BK in blocks of 8
#pragma clang loop unroll(full)
    for (short kk = 0; kk < BK; kk += 8) {
      short2 offset_a =
          transpose_a ? short2(tm + sm, kk + sn) : short2(kk + sn, tm + sm);
      short2 offset_b =
          transpose_b ? short2(kk + sm, tn + sn) : short2(tn + sn, kk + sm);

      const threadgroup T* As__ = As + offset_a.y * lda_tgp + offset_a.x;
      const threadgroup T* Bs__ = Bs + offset_b.y * ldb_tgp + offset_b.x;

      simdgroup_barrier(mem_flags::mem_none);
// Load elements from threadgroup A as simdgroup matrices
#pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
        Asimd[i].thread_elements()[0] = static_cast<AccumType>(As__[0]);
        Asimd[i].thread_elements()[1] = static_cast<AccumType>(As__[jump_a]);
        As__ += simd_stride_a;
      }

      simdgroup_barrier(mem_flags::mem_none);
// Load elements from threadgroup B as simdgroup matrices
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        Bsimd[j].thread_elements()[0] = static_cast<AccumType>(Bs__[0]);
        Bsimd[j].thread_elements()[1] = static_cast<AccumType>(Bs__[jump_b]);
        Bs__ += simd_stride_b;
      }

      simdgroup_barrier(mem_flags::mem_none);
// Multiply and accumulate into result simdgroup matrices
#pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < TN; j++) {
          simdgroup_multiply_accumulate(
              results[i * TN + j], Asimd[i], Bsimd[j], results[i * TN + j]);
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device T* C, const int ldc) const {
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (int j = 0; j < TN; j++) {
        C[(i * TM_stride + sm + tm) * ldc + j * TN_stride + tn + sn] =
            Epilogue::apply(results[i * TN + j].thread_elements()[0]);
        C[(i * TM_stride + sm + tm) * ldc + j * TN_stride + tn + sn + 1] =
            Epilogue::apply(results[i * TN + j].thread_elements()[1]);
      }
    }
  }

  METAL_FUNC void
  store_result_safe(device T* C, const int ldc, short2 dst_tile_dims) const {
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      if (tm + i * TM_stride + sm < dst_tile_dims.y) {
#pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) {
          if (tn + j * TN_stride + sn < dst_tile_dims.x) {
            C[(tm + i * TM_stride + sm) * ldc + tn + j * TN_stride + sn] =
                Epilogue::apply(results[i * TN + j].thread_elements()[0]);
          }

          if (tn + j * TN_stride + sn + 1 < dst_tile_dims.x) {
            C[(tm + i * TM_stride + sm) * ldc + tn + j * TN_stride + sn + 1] =
                Epilogue::apply(results[i * TN + j].thread_elements()[1]);
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<T, AccumType>>
struct GEMMKernel {
  MLX_MTL_CONST short tgp_padding_a = 16 / sizeof(T);
  MLX_MTL_CONST short tgp_padding_b = 16 / sizeof(T);
  MLX_MTL_CONST short tgp_mem_size_a =
      transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a);
  MLX_MTL_CONST short tgp_mem_size_b =
      transpose_b ? BN * (BK + tgp_padding_b) : BK * (BN + tgp_padding_b);
  MLX_MTL_CONST short tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;

  MLX_MTL_CONST short tgp_size = WM * WN * 32;
  MLX_MTL_CONST short vec_size = (BM == 64 && BN == 64) ? 8 : 4;

  using loader_a_t = BlockLoader<
      T,
      BM,
      BK,
      BK,
      vec_size,
      tgp_size,
      transpose_a,
      true,
      tgp_padding_a>;
  using loader_b_t = BlockLoader<
      T,
      BK,
      BN,
      BK,
      vec_size,
      tgp_size,
      transpose_b,
      false,
      tgp_padding_b>;
  using mma_t = BlockMMA<
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      tgp_padding_a,
      tgp_padding_b,
      AccumType,
      Epilogue>;

  /* Main kernel function */
  static METAL_FUNC void run(
      const device T* A [[buffer(0)]],
      const device T* B [[buffer(1)]],
      device T* C [[buffer(2)]],
      const constant int& M [[buffer(3)]],
      const constant int& N [[buffer(4)]],
      const constant int& K [[buffer(5)]],
      const constant int& batch_stride_a [[buffer(6)]],
      const constant int& batch_stride_b [[buffer(7)]],
      const constant int& batch_stride_c [[buffer(8)]],
      threadgroup T* tgp_memory [[threadgroup(0)]],
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    // Pacifying compiler
    (void)lid;

    // Adjust for batch
    A += batch_stride_a * tid.z;
    B += batch_stride_b * tid.z;
    C += batch_stride_c * tid.z;

    // Adjust for transpose
    const int lda_dev = transpose_a ? M : K;
    const int ldb_dev = transpose_b ? K : N;

    // Find block in A, B, C
    const int c_row = tid.y * BM;
    const int c_col = tid.x * BN;

    A += transpose_a ? c_row : c_row * K;
    B += transpose_b ? c_col * K : c_col;
    C += c_row * N + c_col;

    // Prepare threadgroup memory for loading
    threadgroup T* As = tgp_memory;
    threadgroup T* Bs = tgp_memory + tgp_mem_size_a;

    // Prepare threadgroup loading operations
    loader_a_t loader_a(A, lda_dev, As, simd_group_id, simd_lane_id);
    loader_b_t loader_b(B, ldb_dev, Bs, simd_group_id, simd_lane_id);

    // Prepare threadgroup mma operation
    mma_t mma_op(simd_group_id, simd_lane_id);

    ///////////////////////////////////////////////////////////////////////////////
    // MNK aligned loop
    if (MN_aligned && K_aligned) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }

      threadgroup_barrier(mem_flags::mem_none);

      // Store results to device memory
      mma_op.store_result(C, N);
      return;

    }
    ///////////////////////////////////////////////////////////////////////////////
    // MN aligned, K unaligned loop
    else if (MN_aligned && !K_aligned) {
      // Main loop
      int k = 0;
      for (; k + BK <= K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }

      // Loop tail
      threadgroup_barrier(mem_flags::mem_threadgroup);

      loader_a.load_safe(short2(K - k, BM));
      loader_b.load_safe(short2(BN, K - k));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_op.mma(As, Bs);

      // Store results to device memory
      mma_op.store_result(C, N);
      return;

    }
    ///////////////////////////////////////////////////////////////////////////////
    // MNK unaligned loop
    else { // Loop over K - unaligned case

      short2 src_tile_dims(min(BN, N - c_col), min(BM, M - c_row));

      if (src_tile_dims.y == BM && src_tile_dims.x == BN) {
        int k = 0;
        for (; k + BK <= K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          // Load elements into threadgroup
          loader_a.load_unsafe();
          loader_b.load_unsafe();

          threadgroup_barrier(mem_flags::mem_threadgroup);

          // Multiply and accumulate threadgroup elements
          mma_op.mma(As, Bs);

          // Prepare for next iteration
          loader_a.next();
          loader_b.next();
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (k < K) {
          loader_a.load_safe(short2(K - k, BM));
          loader_b.load_safe(short2(BN, K - k));

          threadgroup_barrier(mem_flags::mem_threadgroup);

          mma_op.mma(As, Bs);
        }

        mma_op.store_result(C, N);
        return;

      } else {
        int k = 0;
        for (; k + BK <= K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          // Load elements into threadgroup
          loader_a.load_safe(short2(BK, src_tile_dims.y));
          loader_b.load_safe(short2(src_tile_dims.x, BK));

          threadgroup_barrier(mem_flags::mem_threadgroup);

          // Multiply and accumulate threadgroup elements
          mma_op.mma(As, Bs);

          // Prepare for next iteration
          loader_a.next();
          loader_b.next();
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (k < K) {
          loader_a.load_safe(short2(K - k, src_tile_dims.y));
          loader_b.load_safe(short2(src_tile_dims.x, K - k));

          threadgroup_barrier(mem_flags::mem_threadgroup);

          mma_op.mma(As, Bs);
        }

        threadgroup_barrier(mem_flags::mem_none);
        mma_op.store_result_safe(C, N, src_tile_dims);

        return;
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// END GEMM.H
///////////////////////////////////////////////////////////////////////////////

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

template <typename T,
          int BM,
          int BN,
          int BK,
          int WM,
          int WN,
          bool transpose_a,
          bool transpose_b,
          bool MN_aligned,
          bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm(
    const device T *A [[buffer(0)]],
    const device T *B [[buffer(1)]],
    device T *C [[buffer(2)]],
    const constant int &M [[buffer(3)]],
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &batch_stride_a [[buffer(6)]],
    const constant int &batch_stride_b [[buffer(7)]],
    const constant int &batch_stride_c [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    using gemm_kernel = GEMMKernel<T, BM, BN, BK, WM, WN, transpose_a, transpose_b, MN_aligned, K_aligned>;

    threadgroup T tgp_memory[gemm_kernel::tgp_mem_size];

    gemm_kernel::run(
      A, B, C,
      M, N, K,
      batch_stride_a, batch_stride_b, batch_stride_c,
      tgp_memory,
      simd_lane_id, simd_group_id, tid, lid
    );
}

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel initializations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned) \
  template [[host_name("gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_MN_" #aname "_K_" #kname)]] \
  [[kernel]] void gemm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, mn_aligned, k_aligned>( \
      const device itype *A [[buffer(0)]], \
      const device itype *B [[buffer(1)]], \
      device itype *C [[buffer(2)]], \
      const constant int &M [[buffer(3)]], \
      const constant int &N [[buffer(4)]], \
      const constant int &K [[buffer(5)]], \
      const constant int &batch_stride_a [[buffer(6)]], \
      const constant int &batch_stride_b [[buffer(7)]], \
      const constant int &batch_stride_c [[buffer(8)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 32, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 16, 2, 2)

instantiate_gemm_shapes_helper(float16, half, float16, half);
instantiate_gemm_shapes_helper(float32, float, float32, float);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);

// TODO: Accumulation in different type
