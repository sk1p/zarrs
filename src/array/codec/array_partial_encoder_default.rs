use std::sync::Arc;

use crate::{
    array::{array_bytes::update_array_bytes, ArrayBytes, ArraySize, ChunkRepresentation},
    byte_range::ByteRange,
};

use super::{
    ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits,
    BytesPartialEncoderTraits,
};

/// The default array (chunk) partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct ArrayPartialEncoderDefault<'a> {
    input_handle: Arc<dyn BytesPartialDecoderTraits + 'a>,
    output_handle: Arc<dyn BytesPartialEncoderTraits + 'a>,
    decoded_representation: ChunkRepresentation,
    codec: &'a dyn ArrayToBytesCodecTraits,
}

impl<'a> ArrayPartialEncoderDefault<'a> {
    /// Create a new [`ArrayPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits + 'a>,
        output_handle: Arc<dyn BytesPartialEncoderTraits + 'a>,
        decoded_representation: ChunkRepresentation,
        codec: &'a dyn ArrayToBytesCodecTraits,
    ) -> Self {
        Self {
            input_handle,
            output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl ArrayPartialEncoderTraits for ArrayPartialEncoderDefault<'_> {
    fn partial_encode_opt(
        &self,
        chunk_subsets: &[crate::array_subset::ArraySubset],
        chunk_subsets_bytes: Vec<crate::array::ArrayBytes<'_>>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        // FIXME: Validate

        // Read the entire chunk
        let chunk_shape = self.decoded_representation.shape_u64();
        let chunk_bytes = self.input_handle.decode(options)?;

        // Handle a missing chunk
        let mut chunk_bytes = if let Some(chunk_bytes) = chunk_bytes {
            self.codec
                .decode(chunk_bytes, &self.decoded_representation, options)?
        } else {
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                self.decoded_representation.num_elements(),
            );
            ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value())
        };

        // Validate the bytes
        chunk_bytes.validate(
            self.decoded_representation.num_elements(),
            self.decoded_representation.data_type().size(),
        )?;

        // Update the chunk
        // FIXME: More efficient update for multiple chunk subsets?
        for (chunk_subset, chunk_subset_bytes) in std::iter::zip(chunk_subsets, chunk_subsets_bytes)
        {
            chunk_bytes = update_array_bytes(
                chunk_bytes,
                chunk_shape.clone(), // FIXME
                chunk_subset_bytes,
                chunk_subset,
                self.decoded_representation.data_type().size(),
            );
        }

        let is_fill_value = !options.store_empty_chunks()
            && chunk_bytes.is_fill_value(self.decoded_representation.fill_value());
        if is_fill_value {
            self.output_handle.erase()
        } else {
            // Store the updated chunk
            let chunk_bytes =
                self.codec
                    .encode(chunk_bytes, &self.decoded_representation, options)?;
            self.output_handle.partial_encode_opt(
                &[ByteRange::FromStart(0, Some(chunk_bytes.len() as u64))],
                vec![chunk_bytes],
                options,
            )
        }
    }
}
