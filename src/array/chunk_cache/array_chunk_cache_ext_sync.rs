use std::sync::Arc;

use crate::{
    array::{codec::CodecOptions, ArrayBytes, ArrayError, ElementOwned},
    array_subset::ArraySubset,
    storage::ReadableStorageTraits,
};

use super::{ChunkCache, ChunkCacheType};

#[cfg(doc)]
use crate::array::Array;

/// An [`Array`] extension trait to support reading with a chunk cache.
///
/// Note that these methods never perform partial decoding and always fully decode chunks intersected that are not in the cache.
// TODO: The implementations of this trait would benefit from specialisation. The only thing that differs is retrieve_chunk_opt_cached
pub trait ArrayChunkCacheExt<TStorage: ?Sized + ReadableStorageTraits + 'static, CT: ChunkCacheType>
{
    /// Cached variant of [`retrieve_chunk_opt`](Array::retrieve_chunk_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_opt_cached(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError>;

    /// Cached variant of [`retrieve_chunk_elements_opt`](Array::retrieve_chunk_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_chunk_ndarray_opt`](Array::retrieve_chunk_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Cached variant of [`retrieve_chunks_opt`](Array::retrieve_chunks_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_opt_cached(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Cached variant of [`retrieve_chunks_elements_opt`](Array::retrieve_chunks_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_chunks_ndarray_opt`](Array::retrieve_chunks_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunks_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Cached variant of [`retrieve_chunk_subset_opt`](Array::retrieve_chunk_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_opt_cached(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Cached variant of [`retrieve_chunk_subset_elements_opt`](Array::retrieve_chunk_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_chunk_subset_ndarray_opt`](Array::retrieve_chunk_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_chunk_subset_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Cached variant of [`retrieve_array_subset_opt`](Array::retrieve_array_subset_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_opt_cached(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Cached variant of [`retrieve_array_subset_elements_opt`](Array::retrieve_array_subset_elements_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Cached variant of [`retrieve_array_subset_ndarray_opt`](Array::retrieve_array_subset_ndarray_opt).
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<CT>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;
}
