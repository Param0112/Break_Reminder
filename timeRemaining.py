# time remaining and force exit trigger. 



from os import path, system
import tkinter as tk

self_name = path.basename(__file__)

file = path.expanduser('~\\Auge\\Auge.exe')  # File name

dir_name = path.expanduser('~\\Auge')  # Directory name

if not path.isdir(dir_name):
    system("mkdir " + dir_name)
if not path.isfile(file):
    copy_com = "copy Auge.exe " + dir_name + "\\"
    system(copy_com)
    task = "schtasks /Create /TN \"Auge\" /SC MINUTE /Mo 20 /tr " + "\"" + file + "\"" + " /F 2> nul"
    system(task)


def force_exit():
    print("Force exit triggered.")
    root.quit()  # Close the main Tkinter window to exit the program


def countdown(count):
    label['text'] = 'ON A BREAK, \n TIME REMAINING : {0}s'.format(count)
    if count > 0:
        label.master.after(1000, countdown, count - 1)
    else:
        label.master.quit()


root = tk.Tk()
root.title("Auge App")

label = tk.Label(font=('Times New Roman', '50'), fg='black')
# root.overrideredirect(True)

ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()

hs = hs // 2 - 100
ws = ws // 4
root.geometry('+{0}+{1}'.format(ws, hs))

root.lift()
root.wm_attributes('-topmost', True)
# root.wm_attributes('-disabled', True)
root.wm_attributes('-transparentcolor', 'white')

label.pack()

# Create a function to configure the button's appearance
def configure_button(button):
    button.config(font=('DejaVu Sans', 14), fg='red', bg='black')

# Add a button to force exit the program and apply the button styling
button = tk.Button(root, text="Force exit!", command=force_exit)
configure_button(button)
button.pack(padx=50, pady=20)

countdown(10)

root.mainloop()
