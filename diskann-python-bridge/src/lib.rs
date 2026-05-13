mod build_index;
mod search_index;

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use build_index::{
    build_disk_index,
    BuildDiskIndexParameters,
    DimensionValues,
};

use search_index::{
    search_disk_index,
    SearchDiskIndexParameters,
};

use diskann_disk::data_model::AdHoc;
use diskann_disk::QuantizationType;
use diskann_providers::storage::FileStorageProvider;
use diskann_vector::distance::Metric;

type GraphDataF32Vector = AdHoc<f32>;

fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, ()> {
    if ptr.is_null() {
        return Err(());
    }

    unsafe {
        CStr::from_ptr(ptr)
            .to_str()
            .map_err(|_| ())
    }
}

#[no_mangle]
pub extern "C" fn test_hello() -> c_int {
    println!("hello from rust so");
    0
}

#[no_mangle]
pub extern "C" fn build_index(
    data_path: *const c_char,
    index_path_prefix: *const c_char,
    dim: u32,
    r: u32,
    l: u32,
    num_threads: usize,
    num_pq_chunks: usize,
    ram_limit_gb: f64,
) -> c_int {
    let data_path = match cstr_to_str(data_path) {
        Ok(v) => v,
        Err(_) => return -1,
    };

    let index_path_prefix = match cstr_to_str(index_path_prefix) {
        Ok(v) => v,
        Err(_) => return -2,
    };

    let params = BuildDiskIndexParameters {
        metric: Metric::L2,
        data_path,
        r,
        l,
        index_path_prefix,
        num_threads,
        num_of_pq_chunks: num_pq_chunks,
        index_build_ram_limit_gb: ram_limit_gb,
        build_quantization_type: QuantizationType::FP,
        chunking_parameters: None,
        dim_values: DimensionValues::new(dim as usize, dim as usize),
    };

    match build_disk_index::<GraphDataF32Vector, FileStorageProvider>(
        &FileStorageProvider,
        params,
    ) {
        Ok(_) => {
            println!("build_index done");
            0
        }
        Err(e) => {
            eprintln!("build_index error: {:?}", e);
            -10
        }
    }
}

#[no_mangle]
pub extern "C" fn search_index(
    index_path_prefix: *const c_char,
    query_file: *const c_char,
    top_k: u32,
    l: u32,
    beam_width: usize,
    num_threads: usize,
    out_ids: *mut u32,
    out_capacity: usize,
) -> c_int {
    if out_ids.is_null() {
        return -1;
    }

    let index_path_prefix = match cstr_to_str(index_path_prefix) {
        Ok(v) => v,
        Err(_) => return -2,
    };

    let query_file = match cstr_to_str(query_file) {
        Ok(v) => v,
        Err(_) => return -3,
    };

    let params = SearchDiskIndexParameters {
        metric: Metric::L2,
        index_path_prefix,
        query_file,
        num_threads,
        recall_at: top_k,
        beam_width,
        search_io_limit: usize::MAX,
        l,
        num_nodes_to_cache: 0,
        is_flat_search: false,
    };

    let result = match search_disk_index::<GraphDataF32Vector, FileStorageProvider>(
        &FileStorageProvider,
        params,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("search_index error: {:?}", e);
            return -10;
        }
    };

    let total = result.len() * top_k as usize;

    if total > out_capacity {
        eprintln!(
            "output buffer too small: need {}, capacity {}",
            total, out_capacity
        );
        return -20;
    }

    let mut offset = 0;

    unsafe {
        for row in result {
            for id in row.iter().take(top_k as usize) {
                *out_ids.add(offset) = *id;
                offset += 1;
            }
        }
    }

    offset as c_int
}