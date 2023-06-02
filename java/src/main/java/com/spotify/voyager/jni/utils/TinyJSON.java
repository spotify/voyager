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

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * A dependency-free, super tiny JSON serde class that only
 * supports reading and writing lists of strings.
 */
public class TinyJSON {
    public static List<String> readStringList(InputStream stream) {
        Scanner scanner = new Scanner(stream).useDelimiter("\"");

        List<String> outputList = new ArrayList<>();

        boolean insideString = false;
        while (true) {
            String token = scanner.next();
            if (insideString) {
                outputList.add(token);
                insideString = false;
            } else {
                token = token.trim();
                if (token.equals(",") || token.equals("[")) {
                    insideString = true;
                } else if (token.equals("]") || token.equals("[]")) {
                    break;
                }
            }
        }

        return outputList;
    }

    public static void writeStringList(final Iterable<String> items, final OutputStream stream) throws IOException {
        BufferedWriter output = new BufferedWriter(new OutputStreamWriter(stream));
        output.write("[");
        boolean isFirst = true;
        for (final String item : items) {
            if (!isFirst) {
                output.write(',');
            }
            isFirst = false;
            output.write('"');
            if (item.contains("\\") || item.contains("\"")) {
                throw new IllegalArgumentException(
                        "Voyager string keys may not contain backslashes " +
                                "or double quotes, but found key: " + item
                );
            }
            output.write(item);
            output.write('"');
        }
        output.write("]");
        output.flush();
    }
}
