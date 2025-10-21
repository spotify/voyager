# voyager-node: Node.js Bindings for Spotify's Voyager ANN Library

**Disclaimer:** This project provides Node.js bindings for [Spotify's Voyager](https://github.com/spotify/voyager) approximate nearest neighbor (ANN) search library. It is **not** the official Voyager project, and I am **not affiliated** with Spotify or the Voyager team. The source code is a fork of the original Voyager library, adapted to provide Node.js integration.

## About

This repository enables Node.js developers to leverage the high-performance ANN search capabilities of Voyager. All credit for the original algorithm and C++ implementation goes to the Spotify Voyager team.

## License

This project is distributed under the same license as the original Voyager project. Please refer to the LICENSE file for details.

## Acknowledgements

- [Spotify Voyager](https://github.com/spotify/voyager)
- The Spotify engineering team for their work on the original library

---

## Installation

Install via npm:

```sh
npm install voyager-node
```

## Usage

```js
// ES module
import voyager from "voyager-node";
index = Index.loadIndex(filepath);
// ... use voyager-node API ...
```

or

```js
// CommonJS
const voyager = require("voyager-node");
index = Index.loadIndex(filepath);
// ... use voyager-node API ...
```
