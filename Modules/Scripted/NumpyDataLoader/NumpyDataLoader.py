import logging
import os
import vtk
import slicer
import textwrap
from slicer.ScriptedLoadableModule import *


class NumpyDataLoader(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "NumpyDataLoader"
        parent.categories = ["Testing.TestCases"]
        parent.dependencies = []
        parent.contributors = [
            "Dženan Zukić (Kitware)",
        ]
        parent.helpText = textwrap.dedent(
            """
        Reader for numpy `.npy` files.
        A `vtkMRMLScalarVolumeNode` named after the filename is added to the scene.
        See https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#npy-format
        Only the three dimensions with fastest changing index are loaded.
        """
        )
        parent.acknowledgementText = textwrap.dedent(
            """
        This module is adapted from work done by Steve Pieper to support loading of NIfTI files.
        """
        )
        self.parent = parent


class NumpyDataLoaderWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # Default reload&test widgets are enough.
        # Note that reader and writer is not reloaded.


class NumpyDataLoaderFileReader:
    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return "numpy data file loader"

    def fileType(self):
        return "numpy"

    def extensions(self):
        return ["numpy (*.npy)"]

    def canLoadFile(self, filePath):
        return filePath[-4:] == ".npy"

    def load(self, properties):
        """
        uses properties:
            fileName - path to the .npy file
        """
        try:
            import numpy

            file_path = properties["fileName"]

            # Get node base name from filename
            if "name" in properties.keys():
                base_name = properties["name"]
            else:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
            base_name = slicer.mrmlScene.GenerateUniqueName(base_name)
            scalar_volume = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", base_name
            )

            numpy_image = numpy.load(file_path)
            vtk_image = vtk.vtkImageData()
            shape = numpy_image.shape

            if len(shape) >= 3:
                shape3d = shape[-3:]  # just the last 3 dimensions
                shape3dr = shape3d[::-1]  # reverse it
                vtk_image.SetDimensions(shape3dr)
            elif len(shape) == 2:
                vtk_image.SetDimensions(shape[1], shape[0], 1)  # reverse implicitly
            elif len(shape) == 1:
                vtk_image.SetDimensions(shape[0], 1, 1)  # nothing to reverse
            else:
                raise RuntimeError("Zero dimensional arrays are not supported")

            vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)
            scalar_volume.SetAndObserveImageData(vtk_image)

            node_array = slicer.util.arrayFromVolume(scalar_volume)
            if len(shape) > 3:
                print(f"Warning: only the last 3 dimensions read: {shape3d}")
                print(f"Note: the original shape of {file_path} was {shape}")
                # if the array is 4D or 5D+, we need to select just the first index in each dimension
                l_shape = list(shape)
                for i in range(len(shape) - 3):
                    l_shape[i] = 1
                idx = tuple(slice(0, s, 1) for s in l_shape)
                node_array[:] = numpy_image[idx]
            else:
                node_array[:] = numpy_image
            slicer.util.arrayFromVolumeModified(scalar_volume)

            scalar_volume.CreateDefaultDisplayNodes()

        except Exception as e:
            logging.error("Failed to load numpy data file: " + str(e))
            import traceback

            traceback.print_exc()
            return False

        self.parent.loadedNodes = [scalar_volume.GetID()]
        return True


class NumpyDataLoaderTest(ScriptedLoadableModuleTest):
    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.testWriterReader()
        self.tearDown()
        self.delayDisplay("Testing complete")

    def setUp(self):
        self.tempDir = slicer.util.tempDirectory()
        slicer.mrmlScene.Clear()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tempDir, True)

    def testWriteReadOne(self, a, filename, a3=None):
        """a3 is 3D subset which is expected to be read"""
        import numpy

        a_path = os.path.join(self.tempDir, filename + ".npy")
        numpy.save(a_path, a)

        loaded_node = slicer.util.loadNodeFromFile(a_path, "numpy")
        node_array = slicer.util.arrayFromVolume(loaded_node)
        if a3 is None:
            self.assertTrue(numpy.allclose(a, node_array))
        else:
            self.assertTrue(numpy.allclose(a3, node_array))

    def testWriterReader(self):
        import numpy

        rng = numpy.random.default_rng()

        a5 = rng.random((2, 4, 8, 16, 32))
        self.testWriteReadOne(a5, "a5", a5[0, 0, ...])

        a4 = rng.random((4, 8, 16, 32))
        self.testWriteReadOne(a4, "a4", a4[0, ...])

        a3 = rng.random((8, 16, 32))
        self.testWriteReadOne(a3, "a3")

        a2 = rng.random((16, 32))
        self.testWriteReadOne(a2, "a2")

        a1 = rng.random(32)
        self.testWriteReadOne(a1, "a1")
