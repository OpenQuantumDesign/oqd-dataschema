## Datastore

A [Datastore][oqd_dataschema.datastore.Datastore] represents a HDF5 file of a particular hierarchical structure.

### Hierarchy

```
/
├── group1/
│   └── dataset1
├── group2/
│   ├── dataset2
│   ├── table1
│   └── folder1
└── group3/
    ├── table2
    └── dataset_dict1/
        ├── dataset5
        └── dataset6
```

The top level of [Datastore][oqd_dataschema.datastore.Datastore] contains multiple [Groups](api/group.md)
