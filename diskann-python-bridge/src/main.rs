mod build_index;
mod search_index;

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

fn main() -> anyhow::Result<()> {
    let build_params = BuildDiskIndexParameters {
        metric: Metric::L2,
        data_path: "../test_data/disk_index_search/disk_index_siftsmall_learn_256pts_data.fbin",
        r: 32,
        l: 50,
        index_path_prefix: "siftsmall_index_from_bridge",
        num_threads: 1,
        num_of_pq_chunks: 128,
        index_build_ram_limit_gb: 2.0,
        build_quantization_type: QuantizationType::FP,
        chunking_parameters: None,
        dim_values: DimensionValues::new(128, 128),
    };

    build_disk_index::<GraphDataF32Vector, FileStorageProvider>(
        &FileStorageProvider,
        build_params,
    )?;

    println!("build done");

    let search_params = SearchDiskIndexParameters {
        metric: Metric::L2,
        index_path_prefix: "siftsmall_index_from_bridge",
        query_file: "../test_data/disk_index_search/disk_index_sample_query_10pts.fbin",
        num_threads: 1,
        recall_at: 10,
        beam_width: 4,
        search_io_limit: usize::MAX,
        l: 40,
        num_nodes_to_cache: 0,
        is_flat_search: false,
    };

    let ids = search_disk_index::<GraphDataF32Vector, FileStorageProvider>(
        &FileStorageProvider,
        search_params,
    )?;

    for (qid, row) in ids.iter().enumerate() {
        println!("query {} ids: {:?}", qid, row);
    }

    Ok(())
}