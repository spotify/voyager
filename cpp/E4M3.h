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

static constexpr float ALL_E4M3_VALUES[128] = {
    0.0,          0.0009765625, 0.001953125,  0.0029296875, 0.00390625,
    0.0048828125, 0.005859375,  0.0068359375, 0.015625,     0.017578125,
    0.01953125,   0.021484375,  0.0234375,    0.025390625,  0.02734375,
    0.029296875,  0.03125,      0.03515625,   0.0390625,    0.04296875,
    0.046875,     0.05078125,   0.0546875,    0.05859375,   0.0625,
    0.0703125,    0.078125,     0.0859375,    0.09375,      0.1015625,
    0.109375,     0.1171875,    0.125,        0.140625,     0.15625,
    0.171875,     0.1875,       0.203125,     0.21875,      0.234375,
    0.25,         0.28125,      0.3125,       0.34375,      0.375,
    0.40625,      0.4375,       0.46875,      0.5,          0.5625,
    0.625,        0.6875,       0.75,         0.8125,       0.875,
    0.9375,       1.0,          1.125,        1.25,         1.375,
    1.5,          1.625,        1.75,         1.875,        2.0,
    2.25,         2.5,          2.75,         3.0,          3.25,
    3.5,          3.75,         4.0,          4.5,          5.0,
    5.5,          6.0,          6.5,          7.0,          7.5,
    8.0,          9.0,          10.0,         11.0,         12.0,
    13.0,         14.0,         15.0,         16.0,         18.0,
    20.0,         22.0,         24.0,         26.0,         28.0,
    30.0,         32.0,         36.0,         40.0,         44.0,
    48.0,         52.0,         56.0,         60.0,         64.0,
    72.0,         80.0,         88.0,         96.0,         104.0,
    112.0,        120.0,        128.0,        144.0,        160.0,
    176.0,        192.0,        208.0,        224.0,        240.0,
    256.0,        288.0,        320.0,        352.0,        384.0,
    416.0,        448.0,        NAN};

/**
 * An 8-bit floating point format with a 4-bit exponent and 3-bit mantissa.
 * Inspired by: https://arxiv.org/pdf/2209.05433.pdf
 */
class E4M3 {
public:
  uint8_t sign : 1, exponent : 4, mantissa : 3;

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

  operator float() const {
    // This is implemented with a 512-byte lookup table for speed.
    // Note that the Python tests ensure that this matches the expected logic.
    return (sign ? -1 : 1) * ALL_E4M3_VALUES[(exponent << 3) | mantissa];
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
