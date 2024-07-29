/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 * --
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * -/-/-
 */

#pragma once

#include <cmath>

static constexpr float SMALLEST_POSITIVE_E4M3_VALUE = 0.0009765625f;
static constexpr float ALL_E4M3_VALUES[256] = {
    0,
    -0,
    0.015625,
    -0.015625,
    0.031250,
    -0.031250,
    0.0625,
    -0.0625,
    0.1250,
    -0.1250,
    0.2500,
    -0.2500,
    0.5000,
    -0.5000,
    1,
    -1,
    2,
    -2,
    4,
    -4,
    8,
    -8,
    16,
    -16,
    32,
    -32,
    64,
    -64,
    128,
    -128,
    256,
    -256,
    0.0009765625,
    -0.0009765625,
    0.0175781250,
    -0.0175781250,
    0.03515625,
    -0.03515625,
    0.07031250,
    -0.07031250,
    0.140625,
    -0.140625,
    0.281250,
    -0.281250,
    0.5625,
    -0.5625,
    1.1250,
    -1.1250,
    2.2500,
    -2.2500,
    4.5000,
    -4.5000,
    9,
    -9,
    18,
    -18,
    36,
    -36,
    72,
    -72,
    144,
    -144,
    288,
    -288,
    0.0019531250,
    -0.0019531250,
    0.01953125,
    -0.01953125,
    0.03906250,
    -0.03906250,
    0.078125,
    -0.078125,
    0.156250,
    -0.156250,
    0.3125,
    -0.3125,
    0.6250,
    -0.6250,
    1.2500,
    -1.2500,
    2.5000,
    -2.5000,
    5,
    -5,
    10,
    -10,
    20,
    -20,
    40,
    -40,
    80,
    -80,
    160,
    -160,
    320,
    -320,
    0.0029296875,
    -0.0029296875,
    0.0214843750,
    -0.0214843750,
    0.04296875,
    -0.04296875,
    0.08593750,
    -0.08593750,
    0.171875,
    -0.171875,
    0.343750,
    -0.343750,
    0.6875,
    -0.6875,
    1.3750,
    -1.3750,
    2.7500,
    -2.7500,
    5.5000,
    -5.5000,
    11,
    -11,
    22,
    -22,
    44,
    -44,
    88,
    -88,
    176,
    -176,
    352,
    -352,
    0.00390625,
    -0.00390625,
    0.02343750,
    -0.02343750,
    0.046875,
    -0.046875,
    0.093750,
    -0.093750,
    0.1875,
    -0.1875,
    0.3750,
    -0.3750,
    0.7500,
    -0.7500,
    1.5000,
    -1.5000,
    3,
    -3,
    6,
    -6,
    12,
    -12,
    24,
    -24,
    48,
    -48,
    96,
    -96,
    192,
    -192,
    384,
    -384,
    0.0048828125,
    -0.0048828125,
    0.0253906250,
    -0.0253906250,
    0.05078125,
    -0.05078125,
    0.10156250,
    -0.10156250,
    0.203125,
    -0.203125,
    0.406250,
    -0.406250,
    0.8125,
    -0.8125,
    1.6250,
    -1.6250,
    3.2500,
    -3.2500,
    6.5000,
    -6.5000,
    13,
    -13,
    26,
    -26,
    52,
    -52,
    104,
    -104,
    208,
    -208,
    416,
    -416,
    0.0058593750,
    -0.0058593750,
    0.02734375,
    -0.02734375,
    0.05468750,
    -0.05468750,
    0.109375,
    -0.109375,
    0.218750,
    -0.218750,
    0.4375,
    -0.4375,
    0.8750,
    -0.8750,
    1.7500,
    -1.7500,
    3.5000,
    -3.5000,
    7,
    -7,
    14,
    -14,
    28,
    -28,
    56,
    -56,
    112,
    -112,
    224,
    -224,
    448,
    -448,
    0.0068359375,
    -0.0068359375,
    0.0292968750,
    -0.0292968750,
    0.05859375,
    -0.05859375,
    0.11718750,
    -0.11718750,
    0.234375,
    -0.234375,
    0.468750,
    -0.468750,
    0.9375,
    -0.9375,
    1.8750,
    -1.8750,
    3.7500,
    -3.7500,
    7.5000,
    -7.5000,
    15,
    -15,
    30,
    -30,
    60,
    -60,
    120,
    -120,
    240,
    -240,
    NAN,
    NAN,
};

/**
 * An 8-bit floating point format with a 4-bit exponent and 3-bit mantissa.
 * Inspired by: https://arxiv.org/pdf/2209.05433.pdf
 */
class E4M3 {
public:
  // Note: This actually ends up laid out in a byte as: 0bMMMEEEES
  uint8_t sign : 1;
  uint8_t exponent : 4;
  uint8_t mantissa : 3;

  E4M3() : E4M3(0, 0, 0) {}

  E4M3(uint8_t sign, uint8_t exponent, uint8_t mantissa)
      : sign(sign), exponent(exponent), mantissa(mantissa) {}

  E4M3(uint8_t c)
      : sign(c >> 7), exponent((c >> 3) & 0b1111), mantissa(c & 0b111) {}

  E4M3(float input) {
    if (std::isnan(input) || std::isinf(input)) {
      exponent = 15;
      mantissa = 7;
      return;
    }

    if (input == 0.0) {
      exponent = 0;
      mantissa = 0;
      return;
    }

    // TODO: Don't hard-code these, and instead compute them based on the bit
    // widths above!
    if (input < -448 || input > 448) {
      throw std::domain_error(
          "E4M3 cannot represent values outside of [-448, 448].");
    }

    int originalExponent = ((*((const unsigned int *)&input) &
                             0b01111111100000000000000000000000) >>
                            23);
    int originalMantissa =
        (*((const unsigned int *)&input) & 0b00000000011111111111111111111111);

    sign = input < 0;

    // The "subnormal" number case: where the first bit of the mantissa is
    // not 1.0:
    if (originalExponent - 127 + 7 < 0) {
      exponent = 0;

      int shift = 127 - 7 - (originalExponent & 0b11111111);
      if (shift > 4) {
        // Original value was too small to represent:
        mantissa = 0;
        // Disable rounding by clearing the original mantissa:
        originalMantissa = 0;
      } else {
        originalMantissa |= 1 << 23;
        originalMantissa >>= shift;
        mantissa = originalMantissa >> 20;
      }
    } else if (originalExponent - 127 + 7 > 15) {
      // Should never get here - this should have been caught above!
      throw std::domain_error("E4M3 cannot represent values outside of [-448, "
                              "448] - tried to convert " +
                              std::to_string(input) + ".");
    } else if (originalExponent - 127 + 7 == 0) {
      // Handle the special case where our mantissa wouldn't need to change,
      // except for the fact that an exponent of 0 means the mantissa doesn't
      // have a "1" added to it.
      exponent = 0;
      mantissa = 7;
      if (originalMantissa >> 20 >= 4) {
        // Clear the bit that would cause us to round up:
        originalMantissa &= ~(1 << 19);
        exponent = 1;
        mantissa = 0;
      }
    } else {
      exponent = originalExponent - 127 + 7;
      mantissa = originalMantissa >> 20;
    }

    // If bit 19 is set in the original mantissa,
    // we should round this number to the nearest even...
    int shouldRound = (originalMantissa & (1 << 19));
    // ...unless the rest of the value is non-zero, indicating we must round up.
    int mustRoundUp = shouldRound && (originalMantissa & ((1 << 19) - 1));

    if (shouldRound && !mustRoundUp) {
      // Round to nearest even, per IEEE 754.
      // Use the least significant bit of the mantissa to determine "even":
      if (mantissa & 0b001) {
        // Rounding down (i.e.: doing nothing) would result in an odd result,
        // so we should round up.
        mustRoundUp = true;
      }
    }

    if (mustRoundUp) {
      if (mantissa == 0b111) {
        if (exponent == 0b1111) {
          // Rounding up would push us just outside of the representable range!
          throw std::domain_error(
              "E4M3 cannot represent values outside of [-448, "
              "448] - tried to convert " +
              std::to_string(input) + ".");
        } else {
          exponent++;
          mantissa = 0;
        }
      } else {
        mantissa++;
      }
    }
  }

  inline operator float() const {
    // This is implemented with a 512-byte lookup table for speed.
    // Note that the Python tests ensure that this matches the expected logic.
    return ALL_E4M3_VALUES[*(const uint8_t *)this];
  }

  int8_t effectiveExponent() const { return -7 + exponent; }
  float effectiveMantissa() const {
    if (exponent != 0) {
      return 1.0f + (float)mantissa / 8.0;
    }
    return (float)mantissa / 8.0;
  }

  float operator*(float other) const { return ((float)*this) * other; }
};
