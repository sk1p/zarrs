//! Bytes to bytes codecs.

#[cfg(feature = "blosc")]
pub mod blosc;
#[cfg(feature = "bz2")]
pub mod bz2;
#[cfg(feature = "crc32c")]
pub mod crc32c;
#[cfg(feature = "gzip")]
pub mod gzip;
#[cfg(feature = "lzma")]
pub mod lzma;
#[cfg(feature = "zstd")]
pub mod zstd;
