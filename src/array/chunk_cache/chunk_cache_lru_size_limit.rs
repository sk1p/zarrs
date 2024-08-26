use std::sync::Arc;

use moka::{
    policy::EvictionPolicy,
    sync::{Cache, CacheBuilder},
};

use crate::array::{ArrayBytes, ArrayError, ArrayIndices, RawBytes};

use super::{ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded};
type ChunkIndices = ArrayIndices;

/// A chunk cache with a fixed size capacity.
pub struct ChunkCacheLruSizeLimit<T: ChunkCacheType> {
    cache: Cache<ChunkIndices, Arc<T>>,
}

/// An LRU (least recently used) encoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheEncodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed chunk capacity.
pub type ChunkCacheDecodedLruSizeLimit = ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded>;

impl ChunkCacheLruSizeLimit<ChunkCacheTypeDecoded> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = CacheBuilder::new(capacity)
            .eviction_policy(EvictionPolicy::lru())
            .weigher(|_k, v: &Arc<ArrayBytes<'_>>| u32::try_from(v.size()).unwrap_or(u32::MAX))
            .build();
        Self { cache }
    }
}

impl ChunkCacheLruSizeLimit<ChunkCacheTypeEncoded> {
    /// Create a new [`ChunkCacheLruSizeLimit`] with a capacity in bytes of `capacity`.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        let cache = CacheBuilder::new(capacity)
            .eviction_policy(EvictionPolicy::lru())
            .weigher(|_k, v: &Arc<Option<RawBytes<'_>>>| {
                v.as_ref()
                    .as_ref()
                    .map_or(0, |v| u32::try_from(v.len()).unwrap_or(u32::MAX))
            })
            .build();
        Self { cache }
    }
}

impl<T: ChunkCacheType> ChunkCacheLruSizeLimit<T> {
    /// Return the size of the cache in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.cache.run_pending_tasks();
        usize::try_from(self.cache.weighted_size()).unwrap_or(usize::MAX)
    }
}

macro_rules! impl_ChunkCacheLruSizeLimit {
    ($t:ty) => {
        impl<CT: ChunkCacheType> ChunkCache<CT> for $t {
            fn get(&self, chunk_indices: &[u64]) -> Option<Arc<CT>> {
                self.cache.get(&chunk_indices.to_vec())
            }

            fn insert(&self, chunk_indices: ChunkIndices, chunk: Arc<CT>) {
                self.cache.insert(chunk_indices, chunk);
            }

            fn try_get_or_insert_with<F, E>(
                &self,
                chunk_indices: Vec<u64>,
                f: F,
            ) -> Result<Arc<CT>, Arc<ArrayError>>
            where
                F: FnOnce() -> Result<Arc<CT>, ArrayError>,
            {
                self.cache.try_get_with(chunk_indices, f)
            }

            fn len(&self) -> usize {
                self.cache.run_pending_tasks();
                usize::try_from(self.cache.entry_count()).unwrap()
            }
        }
    };
}

impl_ChunkCacheLruSizeLimit!(ChunkCacheLruSizeLimit<CT>);
impl_ChunkCacheLruSizeLimit!(&ChunkCacheLruSizeLimit<CT>);
