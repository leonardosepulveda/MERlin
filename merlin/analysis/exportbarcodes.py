import pandas

from merlin.core import analysistask


class ExportBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that filters barcodes based on area and mean 
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'columns' not in self.parameters:
            self.parameters['columns'] = ['barcode_id', 'global_x',
                                          'global_y', 'cell_index']
        if 'exclude_blanks' not in self.parameters:
            self.parameters['exclude_blanks'] = True
        # 'parquet' writes barcodes.parquet; 'csv' writes barcodes.csv for
        # pipelines that still read the csv. csv is much larger and slower
        # to write on large experiments.
        if 'format' not in self.parameters:
            self.parameters['format'] = 'parquet'
        if self.parameters['format'] not in ('parquet', 'csv'):
            raise ValueError(
                "ExportBarcodes format must be 'parquet' or 'csv', got %r"
                % self.parameters['format'])

        self.columns = self.parameters['columns']
        self.excludeBlanks = self.parameters['exclude_blanks']

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['filter_task']]

    def _fov_barcodes(self):
        """Yield the barcodes to export, one fov at a time, so memory stays
        at one fov's barcodes. A large experiment has billions of barcodes,
        too many to hold at once."""
        filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])
        barcodeDB = filterTask.get_barcode_database()
        if self.excludeBlanks:
            codingIndexes = filterTask.get_codebook().get_coding_indexes()

        for fov in self.dataSet.get_fovs():
            barcodeData = barcodeDB.get_barcodes(
                fov=fov, columnList=self.columns)

            if self.excludeBlanks:
                barcodeData = barcodeData[
                        barcodeData['barcode_id'].isin(codingIndexes)]

            yield barcodeData

    def _run_analysis(self):
        if self.parameters['format'] == 'csv':
            for i, barcodeData in enumerate(self._fov_barcodes()):
                self.dataSet.save_dataframe_to_csv(
                    barcodeData, 'barcodes', self, append=(i > 0),
                    index=False, header=(i == 0))
            return

        with self.dataSet.open_parquet_chunk_writer(
                'barcodes', self) as writer:
            for barcodeData in self._fov_barcodes():
                writer.write(barcodeData)

        if not writer.wrote_any:
            self.dataSet.save_dataframe_to_parquet(
                pandas.DataFrame(columns=self.columns), 'barcodes', self)
