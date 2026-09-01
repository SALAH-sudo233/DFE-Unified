# External dataset provenance

No CrossDocked or PDBBind dataset content is redistributed in this repository.
The retained DF training configuration expects:

```text
data/crossdocked_pocket10/
data/split_by_name.pt
```

The source server contained a processed CrossDocked LMDB of 3,757,457,408
bytes and a `name2id` file of 26,121,031 bytes. The observed split file was
15,284,527 bytes with SHA-256
`7ba210bc9b8a89034b8e189f26dd055141b0cbcc7aab66898d68e84140ad6e3b`.
These observations identify the inputs but do not establish a public download's
byte-for-byte identity unless its hashes are checked independently.

The upstream Pocket2Mol instructions point to the CrossDocked Pocket10 data used
by the 3D-Generative-SBDD work. Consult the dataset owners' terms and the
upstream project for access. PDBBind data was referenced by evaluation tooling
on the research server but is not included here.

Machine-readable observations are in `crossdocked-manifest.json`.
