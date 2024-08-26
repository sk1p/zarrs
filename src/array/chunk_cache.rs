use std::sync::Arc;

use super::{ArrayBytes, ArrayError, RawBytes};

pub mod array_chunk_cache_ext_decoded_sync;
pub mod array_chunk_cache_ext_encoded_sync;
pub mod array_chunk_cache_ext_sync;
pub mod chunk_cache_lru_chunk_limit;
pub mod chunk_cache_lru_size_limit;

/// The chunk type of an encoded chunk cache.
pub type ChunkCacheTypeEncoded = Option<RawBytes<'static>>;

/// The chunk type of a decoded chunk cache.
pub type ChunkCacheTypeDecoded = ArrayBytes<'static>;

/// A marker trait for a chunk type ([`ChunkCacheTypeEncoded`] or [`ChunkCacheTypeDecoded`]).
pub trait ChunkCacheType: Send + Sync + 'static {}

impl ChunkCacheType for ChunkCacheTypeEncoded {}

impl ChunkCacheType for ChunkCacheTypeDecoded {}

/// Traits for an encoded chunk cache.
pub trait ChunkCache<CT: ChunkCacheType>: Send + Sync {
    /// Retrieve a chunk from the cache. Returns [`None`] if the chunk is not present.
    ///
    /// The chunk cache implementation may modify the cache (e.g. update LRU cache) on retrieval.
    fn get(&self, chunk_indices: &[u64]) -> Option<Arc<CT>>;

    /// Insert a chunk into the cache.
    fn insert(&self, chunk_indices: Vec<u64>, chunk: Arc<CT>);

    /// Get or insert a chunk in the cache.
    ///
    /// Override the default implementation if a chunk offers a more performant implementation.
    ///
    /// # Errors
    /// Returns an error if `f` returns an error.
    fn try_get_or_insert_with<F, E>(
        &self,
        chunk_indices: Vec<u64>,
        f: F,
    ) -> Result<Arc<CT>, Arc<ArrayError>>
    where
        F: FnOnce() -> Result<Arc<CT>, ArrayError>,
    {
        let chunk_indices = chunk_indices.clone();
        if let Some(chunk) = self.get(&chunk_indices) {
            Ok(chunk)
        } else {
            let chunk = f()?;
            self.insert(chunk_indices, chunk.clone());
            Ok(chunk)
        }
    }

    /// Return the number of chunks in the cache.
    #[must_use]
    fn len(&self) -> usize;

    /// Returns true if the cache is empty.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// TODO: AsyncChunkCache
