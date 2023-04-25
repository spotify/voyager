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

package com.spotify.voyager.jni.utils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

public class JniLibExtractor {
  public static String extractBinaries(final String libname) {
    final String mappedLibname = System.mapLibraryName(libname);
    final String libPath = String.format("/%s/%s", platform(), mappedLibname);
    final InputStream library = JniLibExtractor.class.getResourceAsStream(libPath);

    if (library == null) {
      throw new RuntimeException("Could not find JNI library file to load at path: " + libPath);
    }

    try {
      final Path temp = Files.createTempDirectory("").resolve(mappedLibname);
      Files.copy(library, temp);
      temp.toFile().deleteOnExit();
      return temp.toString();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static String platform() {
    String osArch = System.getProperty("os.arch");
    final String osName = System.getProperty("os.name").toLowerCase();

    if (osArch.equals("x86_64") || osArch.equals("amd64")) {
      osArch = "x64";
    }

    if (osName.contains("mac")) {
      return "mac-" + osArch;
    }

    if (osName.contains("linux")) {
      return "linux-" + osArch;
    }

    if (osName.contains("win")) {
      return "win-" + osArch;
    }

    throw new RuntimeException(
        "com.spotify.voyager currently only runs on macOS, Linux, and Windows.");
  }
}
