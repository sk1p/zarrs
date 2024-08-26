use std::sync::Arc;

use moka::{
    policy::EvictionPolicy,
    sync::{Cache, CacheBuilder},
};

use crate::array::{ArrayError, ArrayIndices};

use super::{ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded};

type ChunkIndices = ArrayIndices;

/// A chunk cache with a fixed chunk capacity.
pub struct ChunkCacheLruChunkLimit<T: ChunkCacheType> {
    cache: Cache<ChunkIndices, Arc<T>>,
}

/// An LRU (least recently used) encoded chunk cache with a fixed size in bytes.
pub type ChunkCacheEncodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeEncoded>;

/// An LRU (least recently used) decoded chunk cache with a fixed size in bytes.
pub type ChunkCacheDecodedLruChunkLimit = ChunkCacheLruChunkLimit<ChunkCacheTypeDecoded>;

impl<T: ChunkCacheType> ChunkCacheLruChunkLimit<T> {
    /// Create a new [`ChunkCacheLruChunkLimit`] with a capacity in chunks of `chunk_capacity`.
    #[must_use]
    pub fn new(chunk_capacity: u64) -> Self {
        let cache = CacheBuilder::new(chunk_capacity)
            .eviction_policy(EvictionPolicy::lru())
            .build();
        Self { cache }
    }
}

macro_rules! impl_ChunkCacheLruChunkLimit {
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

impl_ChunkCacheLruChunkLimit!(ChunkCacheLruChunkLimit<CT>);
impl_ChunkCacheLruChunkLimit!(&ChunkCacheLruChunkLimit<CT>);
