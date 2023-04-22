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
    if (exponent == 0b1111 && mantissa == 0b111) {
      return NAN;
    }

    return (sign ? -1 : 1) * powf(2, effectiveExponent()) * effectiveMantissa();
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
