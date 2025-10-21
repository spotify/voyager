# voyager-node

[![npm version](https://img.shields.io/npm/v/voyager-node.svg)](https://www.npmjs.com/package/voyager-node)

Node.js bindings for [Spotify's Voyager](https://github.com/spotify/voyager) approximate nearest neighbor (ANN) search library.

---

**Disclaimer:** This package is not affiliated with Spotify or the Voyager team. It is a community-maintained fork providing Node.js integration for Voyager ANN. All credit for the original algorithm and C++ implementation goes to the Spotify Voyager team.

## Installation

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

See the main repository for detailed documentation and examples.

## License

Distributed under the same license as the original Voyager project. See LICENSE for details.
