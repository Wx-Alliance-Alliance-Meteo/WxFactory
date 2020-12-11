

print_out_enabled = True

def print_out(message):
   global print_out_enabled
   if print_out_enabled:
      print(message)

def print_to_file(filename, message):
   global print_out_enabled
   if print_out_enabled and filename is not None:
      with open(filename, 'a') as out_file:
         out_file.write(message + '\n')

def enable_print_out(status = True):
   global print_out_enabled
   print_out_enabled = status

