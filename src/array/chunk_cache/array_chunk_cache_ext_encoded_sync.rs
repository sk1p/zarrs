use std::{borrow::Cow, sync::Arc};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;

use crate::{
    array::{
        array_bytes::{merge_chunks_vlen, update_bytes_flen},
        codec::{ArrayToBytesCodecTraits, CodecOptions},
        concurrency::concurrency_chunks_and_codec,
        Array, ArrayBytes, ArrayChunkCacheExt, ArrayError, ArraySize, DataTypeSize, ElementOwned,
        UnsafeCellSlice,
    },
    array_subset::ArraySubset,
    storage::{ReadableStorageTraits, StorageError},
};

use super::{ChunkCache, ChunkCacheTypeEncoded};

impl<TStorage: ?Sized + ReadableStorageTraits + 'static>
    ArrayChunkCacheExt<TStorage, ChunkCacheTypeEncoded> for Array<TStorage>
{
    fn retrieve_chunk_opt_cached(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        let chunk_encoded = cache
            .try_get_or_insert_with::<_, ArrayError>(chunk_indices.to_vec(), || {
                Ok(Arc::new(
                    self.retrieve_encoded_chunk(chunk_indices)?.map(Cow::Owned),
                ))
            })
            .map_err(|err| {
                // moka returns an Arc'd error, unwrap it noting that ArrayError is not cloneable
                Arc::try_unwrap(err).unwrap_or_else(|err| {
                    ArrayError::StorageError(StorageError::from(err.to_string()))
                })
            })?;
        // let chunk_encoded = if let Some(chunk) = cache.get(chunk_indices) {
        //     chunk
        // } else {
        //     let chunk = self.retrieve_encoded_chunk(chunk_indices)?.map(Cow::Owned);
        //     let chunk = Arc::new(chunk);
        //     cache.insert(chunk_indices.to_vec(), chunk.clone());
        //     chunk
        // };

        if let Some(chunk_encoded) = chunk_encoded.as_ref() {
            let chunk_representation = self.chunk_array_representation(chunk_indices)?;
            let bytes = self
                .codecs()
                .decode(Cow::Borrowed(chunk_encoded), &chunk_representation, options)
                .map_err(ArrayError::CodecError)?;
            bytes.validate(
                chunk_representation.num_elements(),
                chunk_representation.data_type().size(),
            )?;
            Ok(Arc::new(bytes.into_owned()))
        } else {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let array_size =
                ArraySize::new(self.data_type().size(), chunk_shape.num_elements_u64());
            Ok(Arc::new(ArrayBytes::new_fill_value(
                array_size,
                self.fill_value(),
            )))
        }
    }

    fn retrieve_chunk_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            Arc::unwrap_or_clone(self.retrieve_chunk_opt_cached(cache, chunk_indices, options)?),
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_chunk_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let shape = self
            .chunk_grid()
            .chunk_shape_u64(chunk_indices, self.shape())?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?;
        crate::array::elements_to_ndarray(
            &shape,
            self.retrieve_chunk_elements_opt_cached::<T>(cache, chunk_indices, options)?,
        )
    }

    fn retrieve_chunks_opt_cached(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if chunks.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                chunks.clone(),
                self.shape().to_vec(),
            ));
        }

        let array_subset = self.chunks_subset(chunks)?;
        self.retrieve_array_subset_opt_cached(cache, &array_subset, options)
    }

    fn retrieve_chunks_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunks_opt_cached(cache, chunks, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_chunks_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        let elements = self.retrieve_chunks_elements_opt_cached::<T>(cache, chunks, options)?;
        crate::array::elements_to_ndarray(array_subset.shape(), elements)
    }

    fn retrieve_chunk_subset_opt_cached(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let chunk_bytes = self.retrieve_chunk_opt_cached(cache, chunk_indices, options)?;
        let chunk_subset_bytes = chunk_bytes
            .extract_array_subset(chunk_subset, chunk_subset.shape(), self.data_type())?
            .into_owned();
        Ok(chunk_subset_bytes)
    }

    fn retrieve_chunk_subset_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_chunk_subset_opt_cached(cache, chunk_indices, chunk_subset, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_chunk_subset_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        chunk_indices: &[u64],
        chunk_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements = self.retrieve_chunk_subset_elements_opt_cached::<T>(
            cache,
            chunk_indices,
            chunk_subset,
            options,
        )?;
        crate::array::elements_to_ndarray(chunk_subset.shape(), elements)
    }

    #[allow(clippy::too_many_lines)]
    fn retrieve_array_subset_opt_cached(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if array_subset.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.clone(),
                self.shape().to_vec(),
            ));
        }

        // Find the chunks intersecting this array subset
        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.clone(),
                self.shape().to_vec(),
            ));
        };

        let chunk_representation0 =
            self.chunk_array_representation(&vec![0; self.dimensionality()])?;

        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let array_size =
                    ArraySize::new(self.data_type().size(), array_subset.num_elements());
                Ok(ArrayBytes::new_fill_value(array_size, self.fill_value()))
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if &chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    Ok(Arc::unwrap_or_clone(self.retrieve_chunk_opt_cached(
                        cache,
                        chunk_indices,
                        options,
                    )?))
                } else {
                    let array_subset_in_chunk_subset =
                        unsafe { array_subset.relative_to_unchecked(chunk_subset.start()) };
                    self.retrieve_chunk_subset_opt_cached(
                        cache,
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        options,
                    )
                }
            }
            _ => {
                // Calculate chunk/codec concurrency
                let num_chunks = chunks.num_elements_usize();
                let codec_concurrency =
                    self.recommended_codec_concurrency(&chunk_representation0)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                // Retrieve chunks
                let indices = chunks.indices();
                let chunk_bytes_and_subsets =
                    iter_concurrent_limit!(chunk_concurrent_limit, indices, map, |chunk_indices| {
                        let chunk_subset = self.chunk_subset(&chunk_indices)?;
                        self.retrieve_chunk_opt_cached(cache, &chunk_indices, &options)
                            .map(|bytes| (bytes, chunk_subset))
                    })
                    .collect::<Result<Vec<_>, ArrayError>>()?;

                // Merge
                match self.data_type().size() {
                    DataTypeSize::Variable => {
                        // Arc<ArrayBytes> -> ArrayBytes (not copied, but a bit wasteful, change merge_chunks_vlen?)
                        let chunk_bytes_and_subsets = chunk_bytes_and_subsets
                            .iter()
                            .map(|(chunk_bytes, chunk_subset)| {
                                (ArrayBytes::clone(chunk_bytes), chunk_subset.clone())
                            })
                            .collect();
                        Ok(merge_chunks_vlen(
                            chunk_bytes_and_subsets,
                            array_subset.shape(),
                        )?)
                    }
                    DataTypeSize::Fixed(data_type_size) => {
                        // Allocate the output
                        let size_output = array_subset.num_elements_usize() * data_type_size;
                        let mut output = Vec::with_capacity(size_output);

                        {
                            let output =
                                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
                            let update_output = |(chunk_subset_bytes, chunk_subset): (
                                Arc<ArrayBytes>,
                                ArraySubset,
                            )| {
                                // Extract the overlapping bytes
                                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                                let chunk_subset_bytes = if chunk_subset_overlap == chunk_subset {
                                    chunk_subset_bytes
                                } else {
                                    Arc::new(chunk_subset_bytes.extract_array_subset(
                                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                                        chunk_subset.shape(),
                                        self.data_type(),
                                    )?)
                                };

                                let fixed = match chunk_subset_bytes.as_ref() {
                                    ArrayBytes::Fixed(fixed) => fixed,
                                    ArrayBytes::Variable(_, _) => unreachable!(),
                                };

                                update_bytes_flen(
                                    unsafe { output.get() },
                                    array_subset.shape(),
                                    fixed,
                                    &chunk_subset_overlap.relative_to(array_subset.start())?,
                                    data_type_size,
                                );
                                Ok::<_, ArrayError>(())
                            };
                            iter_concurrent_limit!(
                                chunk_concurrent_limit,
                                chunk_bytes_and_subsets,
                                try_for_each,
                                update_output
                            )?;
                        }
                        unsafe { output.set_len(size_output) };
                        Ok(ArrayBytes::from(output))
                    }
                }
            }
        }
    }

    fn retrieve_array_subset_elements_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_array_subset_opt_cached(cache, array_subset, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_array_subset_ndarray_opt_cached<T: ElementOwned>(
        &self,
        cache: &impl ChunkCache<ChunkCacheTypeEncoded>,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements =
            self.retrieve_array_subset_elements_opt_cached::<T>(cache, array_subset, options)?;
        crate::array::elements_to_ndarray(array_subset.shape(), elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::{
        array::{
            ArrayBuilder, ChunkCacheEncodedLruChunkLimit, ChunkCacheEncodedLruSizeLimit, DataType,
            FillValue,
        },
        array_subset::ArraySubset,
        storage::{storage_transformer::PerformanceMetricsStorageTransformer, store::MemoryStore},
    };

    #[test]
    fn array_chunk_cache_chunks() {
        use crate::storage::storage_transformer::StorageTransformerExtension;

        let performance_metrics = Arc::new(PerformanceMetricsStorageTransformer::new());
        let store = Arc::new(MemoryStore::default());
        let store = performance_metrics
            .clone()
            .create_readable_writable_transformer(store);
        let builder = ArrayBuilder::new(
            vec![8, 8], // array shape
            DataType::UInt8,
            vec![4, 4].try_into().unwrap(), // regular chunk shape
            FillValue::from(0u8),
        );
        let array = builder.build(store, "/").unwrap();

        let data: Vec<u8> = (0..array.shape().into_iter().product())
            .map(|i| i as u8)
            .collect();
        array
            .store_array_subset_elements(
                &ArraySubset::new_with_shape(array.shape().to_vec()),
                &data,
            )
            .unwrap();

        let cache = ChunkCacheEncodedLruChunkLimit::new(2);

        assert_eq!(performance_metrics.reads(), 0);
        assert!(cache.is_empty());
        assert_eq!(
            array
                .retrieve_array_subset_opt_cached(
                    &cache,
                    &ArraySubset::new_with_ranges(&[3..5, 0..4]),
                    &CodecOptions::default()
                )
                .unwrap(),
            vec![24, 25, 26, 27, 32, 33, 34, 35,].into()
        );
        assert_eq!(performance_metrics.reads(), 2);
        assert_eq!(cache.len(), 2);

        // Retrieve a chunk in cache
        assert_eq!(
            array
                .retrieve_chunk_opt_cached(&cache, &[0, 0], &CodecOptions::default())
                .unwrap(),
            Arc::new(vec![0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27].into())
        );
        assert_eq!(performance_metrics.reads(), 2);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&[0, 0]).is_some());
        assert!(cache.get(&[1, 0]).is_some());

        // Retrieve a chunk not in cache
        assert_eq!(
            array
                .retrieve_chunk_opt_cached(&cache, &[0, 1], &CodecOptions::default())
                .unwrap(),
            Arc::new(vec![4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31].into())
        );
        assert_eq!(performance_metrics.reads(), 3);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&[0, 1]).is_some());
        assert!(cache.get(&[0, 0]).is_none() || cache.get(&[1, 0]).is_none());
    }

    #[test]
    fn array_chunk_cache_size() {
        use crate::storage::storage_transformer::StorageTransformerExtension;

        let performance_metrics = Arc::new(PerformanceMetricsStorageTransformer::new());
        let store = Arc::new(MemoryStore::default());
        let store = performance_metrics
            .clone()
            .create_readable_writable_transformer(store);
        let builder = ArrayBuilder::new(
            vec![8, 8], // array shape
            DataType::UInt8,
            vec![4, 4].try_into().unwrap(), // regular chunk shape
            FillValue::from(0u8),
        );
        let array = builder.build(store, "/").unwrap();

        let data: Vec<u8> = (0..array.shape().into_iter().product())
            .map(|i| i as u8)
            .collect();
        array
            .store_array_subset_elements(
                &ArraySubset::new_with_shape(array.shape().to_vec()),
                &data,
            )
            .unwrap();

        // Create a cache with a size limit equivalent to 2 chunks
        let chunk_size = 4 * 4 * size_of::<u8>();
        let cache = ChunkCacheEncodedLruSizeLimit::new(2 * chunk_size as u64);

        assert_eq!(performance_metrics.reads(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.size(), 0);
        assert_eq!(
            array
                .retrieve_array_subset_opt_cached(
                    &cache,
                    &ArraySubset::new_with_ranges(&[3..5, 0..4]),
                    &CodecOptions::default()
                )
                .unwrap(),
            vec![24, 25, 26, 27, 32, 33, 34, 35,].into()
        );
        assert_eq!(performance_metrics.reads(), 2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), chunk_size * 2);

        // Retrieve a chunk in cache
        assert_eq!(
            array
                .retrieve_chunk_opt_cached(&cache, &[0, 0], &CodecOptions::default())
                .unwrap(),
            Arc::new(vec![0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27].into())
        );
        assert_eq!(performance_metrics.reads(), 2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), chunk_size * 2);
        assert!(cache.get(&[0, 0]).is_some());
        assert!(cache.get(&[1, 0]).is_some());

        // Retrieve a chunk not in cache
        assert_eq!(
            array
                .retrieve_chunk_opt_cached(&cache, &[0, 1], &CodecOptions::default())
                .unwrap(),
            Arc::new(vec![4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31].into())
        );
        assert_eq!(performance_metrics.reads(), 3);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size(), chunk_size * 2);
        assert!(cache.get(&[0, 1]).is_some());
        assert!(cache.get(&[0, 0]).is_none() || cache.get(&vec![1, 0]).is_none());
    }
}
