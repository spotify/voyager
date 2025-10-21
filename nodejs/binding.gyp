{
  "targets": [
    {
      "target_name": "voyager-node",
      "cflags!": [
        "-fno-exceptions"
      ],
      "cflags_cc!": [
        "-fno-exceptions"
      ],
      "cflags_cc": [
        "-std=c++17",
        "-O3"
      ],
      "sources": [
        "src/voyager-node.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "voyager_src",
        "../cpp/src"
      ],
      "defines": [
        "NAPI_CPP_EXCEPTIONS"
      ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.15",
        "OTHER_CFLAGS": [
          "-O3"
        ]
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "AdditionalOptions": [
            "/std:c++17"
          ]
        }
      },
      "conditions": [
        [
          "OS=='linux'",
          {
            "cflags_cc": [
              "-std=c++17",
              "-O3",
              "-pthread"
            ],
            "ldflags": [
              "-pthread"
            ]
          }
        ],
        [
          "OS=='mac'",
          {
            "cflags_cc": [
              "-std=c++17",
              "-O3"
            ]
          }
        ]
      ]
    }
  ]
}
