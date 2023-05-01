from csv import writer
import sys

def append_list_as_row(list_of_elem,file_name="mps_throughput.csv"):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)



if (sys.argv[6]=='cuda'):
	append_list_as_row(sys.argv[1:6],"mps_throughput_cuda.csv")
elif (sys.argv[6]=='cpu'):
	append_list_as_row(sys.argv[1:6],"mps_throughput_cpu.csv")
