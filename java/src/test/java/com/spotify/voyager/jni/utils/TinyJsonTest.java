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

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TinyJsonTest {
    @Test
    public void testSerializeList() throws IOException {
        final ByteArrayOutputStream os = new ByteArrayOutputStream();
        TinyJson.writeStringList(Arrays.asList("a", "b", "c"), os);
        assertEquals("[\"a\",\"b\",\"c\"]", os.toString());
    }

    @Test
    public void testDeserializeList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\",\"b\",\"c\"]".getBytes(UTF_8));
        List<String> result = TinyJson.readStringList(is);
        assertEquals(Arrays.asList("a", "b", "c"), result);
    }

    @Test
    public void testDeserializeListWithWhitespace() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\", \"b\",\n\n \"c\"]".getBytes(UTF_8));
        List<String> result = TinyJson.readStringList(is);
        assertEquals(Arrays.asList("a", "b", "c"), result);
    }

    @Test
    public void testDeserializeNonList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("{}".getBytes(UTF_8));
        try {
            TinyJson.readStringList(is);
            fail();
        } catch (IllegalArgumentException ignored) {
        }
    }

    @Test
    public void testDeserializeNonStringList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\", 1]".getBytes(UTF_8));
        try {
            TinyJson.readStringList(is);
            fail();
        } catch (IllegalArgumentException ignored) {
        }
    }

    @Test
    public void testDeserializeInvalidList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\"}".getBytes(UTF_8));
        try {
            TinyJson.readStringList(is);
            fail();
        } catch (IllegalArgumentException ignored) {
        }
    }
}
