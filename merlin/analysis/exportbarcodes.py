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

        self.columns = self.parameters['columns']
        self.excludeBlanks = self.parameters['exclude_blanks']

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['filter_task']]

    def _run_analysis(self):
        filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])        

        barcodeDB = filterTask.get_barcode_database()
        if self.excludeBlanks:
            codingIndexes = filterTask.get_codebook().get_coding_indexes()

        # Written one fov at a time as parquet row groups, so memory stays
        # at one fov's barcodes. A large experiment has billions of
        # barcodes, too many to hold at once or to write as csv in time.
        with self.dataSet.open_parquet_chunk_writer(
                'barcodes', self) as writer:
            for fov in self.dataSet.get_fovs():
                barcodeData = barcodeDB.get_barcodes(
                    fov=fov, columnList=self.columns)

                if self.excludeBlanks:
                    barcodeData = barcodeData[
                            barcodeData['barcode_id'].isin(codingIndexes)]

                writer.write(barcodeData)

        if not writer.wrote_any:
            self.dataSet.save_dataframe_to_parquet(
                pandas.DataFrame(columns=self.columns), 'barcodes', self)
