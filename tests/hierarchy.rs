use std::sync::Arc;

use zarrs::{
    group::Group,
    metadata::v3::group::ConsolidatedMetadata,
    node::{Node, NodePath},
    storage::store::FilesystemStore,
};

#[test]
fn hierarchy_tree() {
    let store = Arc::new(
        FilesystemStore::new("./tests/data/hierarchy.zarr")
            .unwrap()
            .sorted(),
    );
    let node = Node::open(&store, "/").unwrap();
    let tree = node.hierarchy_tree();
    println!("{:?}", tree);
    assert_eq!(
        tree,
        "/
  a
    baz [10000, 1000] float64
    foo [10000, 1000] float64
  b
"
    );
}

#[test]
fn consolidated_metadata() {
    let store = Arc::new(
        FilesystemStore::new("./tests/data/hierarchy.zarr")
            .unwrap()
            .sorted(),
    );
    let node = Node::open(&store, "/").unwrap();
    let consolidated_metadata = node.consolidate_metadata().unwrap();

    for node_path in ["/a/baz", "/a/foo", "/b"] {
        let consolidated = consolidated_metadata
            .get(&NodePath::new(node_path).unwrap())
            .unwrap();
        let actual = Node::open(&store, node_path).unwrap();
        assert_eq!(consolidated, actual.metadata());
    }

    let mut group = Group::open(store, "/").unwrap();
    assert!(group.consolidated_metadata().is_none());
    group.set_consolidated_metadata(Some(ConsolidatedMetadata {
        metadata: consolidated_metadata,
        ..Default::default()
    }));
    assert!(group.consolidated_metadata().is_some());
}
