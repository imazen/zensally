//! Public-API surface snapshots for the PARENT package (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates([
            "zensally",
            "zensally-tract",
            "zensally-zentract",
            "zensally-depth",
            "zensally-micro",
        ])
        .skip_packaging_check("zensally-depth")
        .skip_packaging_check("zensally-micro")
        .run();
}
