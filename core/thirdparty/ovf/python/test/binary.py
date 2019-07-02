import os
import sys

ovf_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), ".."))
sys.path.insert(0, ovf_py_dir)

from ovf import ovf

import numpy as np

import unittest

##########

class TestState(unittest.TestCase):
    def test_write_bin(self):
        print("----- ovf test writing")
        with ovf.ovf_file("testfile_py_bin.ovf") as ovf_file:
            data = np.zeros((2, 2, 1, 3), dtype='d')
            data[0,1,0,:] = [3.0, 2.0, 1.0]
            segment = ovf.ovf_segment(
                title="python write test",
                comment="more details in this comment...",
                valuedim=3,
                n_cells=[2,2,1])
            success = ovf_file.write_segment(segment, data, fileformat=ovf.FILEFORMAT_BIN)
            if success != ovf.OK:
                print("write_segment failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
            data[0,1,0,:] = [4.0, 5.0, 6.0]
            segment.title = "python append test".encode('utf-8')
            success = ovf_file.append_segment(segment, data, fileformat=ovf.FILEFORMAT_BIN)
            if success != ovf.OK:
                print("append_segment failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
        print("----- ovf test writing done")

        print("----- ovf test reading")
        with ovf.ovf_file("testfile_py_bin.ovf") as ovf_file:
            print("found:      ", ovf_file.found)
            print("is_ovf:     ", ovf_file.is_ovf)
            print("n_segments: ", ovf_file.n_segments)
            segment = ovf.ovf_segment()
            success = ovf_file.read_segment_header(0, segment)
            if success != ovf.OK:
                print("read_segment_header failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
            data_shape = (segment.n_cells[0], segment.n_cells[1], segment.n_cells[2], 3)
            data = np.zeros(data_shape, dtype='f')
            print("data shape: ", data_shape)
            success = ovf_file.read_segment_data(0, segment, data)
            if success != ovf.OK:
                print("read_segment_data failed: ", ovf_file.get_latest_message())
            print("first segment:  ", data[0,1,0,:])
            self.assertTrue( success == ovf.OK )

            success = ovf_file.read_segment_header(1, segment)
            if success != ovf.OK:
                print("read_segment_header failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
            data_shape = (segment.n_cells[0], segment.n_cells[1], segment.n_cells[2], 3)
            data = np.zeros(data_shape, dtype='d')
            success = ovf_file.read_segment_data(1, segment, data)
            if success != ovf.OK:
                print("read_segment_data failed: ", ovf_file.get_latest_message())
            print("second segment: ", data[0,1,0,:])
            self.assertTrue( success == ovf.OK )
        print("----- ovf test reading done")


    def test_write_mixed(self):
        print("----- ovf test writing")
        with ovf.ovf_file("testfile_py_mixed.ovf") as ovf_file:
            data = np.zeros((2, 2, 1, 3), dtype='d')
            data[0,1,0,:] = [3.0, 2.0, 1.0]
            segment = ovf.ovf_segment(
                title="python write test",
                comment="more details in this comment...",
                valuedim=3,
                n_cells=[2,2,1])
            success = ovf_file.write_segment(segment, data, fileformat=ovf.FILEFORMAT_BIN)
            if success != ovf.OK:
                print("write_segment failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
            data[0,1,0,:] = [4.0, 5.0, 6.0]
            segment.title = "python append test".encode('utf-8')
            success = ovf_file.append_segment(segment, data, fileformat=ovf.FILEFORMAT_CSV)
            if success != ovf.OK:
                print("append_segment failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
        print("----- ovf test writing done")

        print("----- ovf test reading")
        with ovf.ovf_file("testfile_py_mixed.ovf") as ovf_file:
            print("found:      ", ovf_file.found)
            print("is_ovf:     ", ovf_file.is_ovf)
            print("n_segments: ", ovf_file.n_segments)
            segment = ovf.ovf_segment()
            success = ovf_file.read_segment_header(0, segment)
            if success != ovf.OK:
                print("read_segment_header failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
            data_shape = (segment.n_cells[0], segment.n_cells[1], segment.n_cells[2], 3)
            data = np.zeros(data_shape, dtype='f')
            print("data shape: ", data_shape)
            success = ovf_file.read_segment_data(0, segment, data)
            if success != ovf.OK:
                print("read_segment_data failed: ", ovf_file.get_latest_message())
            print("first segment:  ", data[0,1,0,:])
            self.assertTrue( success == ovf.OK )

            success = ovf_file.read_segment_header(1, segment)
            if success != ovf.OK:
                print("read_segment_header failed: ", ovf_file.get_latest_message())
            self.assertTrue( success == ovf.OK )
            data_shape = (segment.n_cells[0], segment.n_cells[1], segment.n_cells[2], 3)
            data = np.zeros(data_shape, dtype='d')
            success = ovf_file.read_segment_data(1, segment, data)
            if success != ovf.OK:
                print("read_segment_data failed: ", ovf_file.get_latest_message())
            print("second segment: ", data[0,1,0,:])
            self.assertTrue( success == ovf.OK )
        print("----- ovf test reading done")

#########


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestState)
    success = unittest.TextTestRunner().run(suite).wasSuccessful()
    sys.exit(not success)