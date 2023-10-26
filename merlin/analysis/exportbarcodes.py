from merlin.core import analysistask


class ExportBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that filters barcodes based on area and mean 
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'columns' not in self.parameters:
            # Available barcode information:
            #
            # 'barcode_id', 'fov', 'cell_index', 
            # 'mean_intensity', 'max_intensity', 'area',
            # 'mean_distance', 'min_distance', 
            # 'x', 'y', 'z', 'global_x', 'global_y','global_z', 
            # 'intensity_0', 'intensity_1', ... 'intensity_Nbits',
            #
            # TODO: current implementation does not provide cell_index

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

        barcodeData = filterTask.get_barcode_database().get_barcodes(
            columnList=self.columns)

        if self.excludeBlanks:
            codebook = filterTask.get_codebook()
            barcodeData = barcodeData[
                    barcodeData['barcode_id'].isin(
                        codebook.get_coding_indexes())]

        self.dataSet.save_dataframe_to_csv(barcodeData, 'barcodes', self,
                                           index=False)
