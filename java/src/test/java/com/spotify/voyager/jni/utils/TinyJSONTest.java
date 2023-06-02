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

public class TinyJSONTest {
    @Test
    public void testSerializeList() throws IOException {
        final ByteArrayOutputStream os = new ByteArrayOutputStream();
        TinyJSON.writeStringList(Arrays.asList("a", "b", "c"), os);
        assertEquals("[\"a\",\"b\",\"c\"]", os.toString());
    }

    @Test
    public void testDeserializeList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\",\"b\",\"c\"]".getBytes(UTF_8));
        List<String> result = TinyJSON.readStringList(is);
        assertEquals(Arrays.asList("a", "b", "c"), result);
    }

    @Test
    public void testDeserializeListWithWhitespace() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\", \"b\",\n\n \"c\"]".getBytes(UTF_8));
        List<String> result = TinyJSON.readStringList(is);
        assertEquals(Arrays.asList("a", "b", "c"), result);
    }

    @Test
    public void testDeserializeNonList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("{}".getBytes(UTF_8));
        try {
            TinyJSON.readStringList(is);
            fail();
        } catch (IllegalArgumentException ignored) {
        }
    }

    @Test
    public void testDeserializeNonStringList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\", 1]".getBytes(UTF_8));
        try {
            TinyJSON.readStringList(is);
            fail();
        } catch (IllegalArgumentException ignored) {
        }
    }

    @Test
    public void testDeserializeInvalidList() {
        final ByteArrayInputStream is = new ByteArrayInputStream("[\"a\"}".getBytes(UTF_8));
        try {
            TinyJSON.readStringList(is);
            fail();
        } catch (IllegalArgumentException ignored) {
        }
    }
}
