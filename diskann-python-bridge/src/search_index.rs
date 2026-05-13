use anyhow::Result;

use diskann_disk::{
    data_model::{CachingStrategy, GraphDataType},
    search::provider::{
        disk_provider::DiskIndexSearcher,
        disk_vertex_provider_factory::DiskVertexProviderFactory,
    },
    storage::disk_index_reader::DiskIndexReader,
    utils::AlignedFileReaderFactory,
};

use diskann_providers::storage::{
    get_compressed_pq_file,
    get_disk_index_file,
    get_pq_pivot_file,
    StorageReadProvider,
};

use diskann_utils::io::read_bin;
use diskann_vector::distance::Metric;

pub struct SearchDiskIndexParameters<'a> {
    pub metric: Metric,
    pub index_path_prefix: &'a str,
    pub query_file: &'a str,
    pub num_threads: usize,
    pub recall_at: u32,
    pub beam_width: usize,
    pub search_io_limit: usize,
    pub l: u32,
    pub num_nodes_to_cache: usize,
    pub is_flat_search: bool,
}

pub fn search_disk_index<Data, StorageType>(
    storage_provider: &StorageType,
    parameters: SearchDiskIndexParameters,
) -> Result<Vec<Vec<u32>>>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageType: StorageReadProvider,
{
    let queries = read_bin::<Data::VectorDataType>(
        &mut storage_provider.open_reader(parameters.query_file)?,
    )?;

    let query_num = queries.nrows();

    let index_reader = DiskIndexReader::<<Data as GraphDataType>::VectorDataType>::new(
        get_pq_pivot_file(parameters.index_path_prefix),
        get_compressed_pq_file(parameters.index_path_prefix),
        storage_provider,
    )?;

    let caching_strategy = if parameters.num_nodes_to_cache > 0 {
        CachingStrategy::StaticCacheWithBfsNodes(parameters.num_nodes_to_cache)
    } else {
        CachingStrategy::None
    };

    let reader_factory =
        AlignedFileReaderFactory::new(get_disk_index_file(parameters.index_path_prefix));

    let vertex_provider_factory =
        DiskVertexProviderFactory::<Data, _>::new(reader_factory, caching_strategy)?;

    let searcher = DiskIndexSearcher::<Data, _>::new(
        parameters.num_threads,
        parameters.search_io_limit,
        &index_reader,
        vertex_provider_factory,
        parameters.metric,
        None,
    )?;

    let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(query_num);

    for qid in 0..query_num {
        let query = queries.row(qid);

        let result = searcher.search(
            query,
            parameters.recall_at,
            parameters.l,
            Some(parameters.beam_width),
            None,
            parameters.is_flat_search,
        )?;

        let ids: Vec<u32> = result
            .results
            .iter()
            .take(parameters.recall_at as usize)
            .map(|item| item.vertex_id)
            .collect();

        all_ids.push(ids);
    }

    Ok(all_ids)
}