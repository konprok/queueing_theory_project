import tkinter as tk
from tkinter import messagebox
from config import CONFIG, validate_config
from simulation import HospitalNetwork
from analysis import run_bcmp_analysis
import numpy as np

def validate_with_ranges(config):
    errors = []

    server_limits = {
        'registration_servers': (1, 10),
        'admissions_servers': (1, 10),
        'gynecology_servers': (1, 20),
        'predelivery_servers': (1, 30),
        'delivery_servers': (1, 10),
        'icu_servers': (1, 5),
        'postpartum_servers': (1, 25)
    }

    for key, (min_val, max_val) in server_limits.items():
        if not (min_val <= config[key] <= max_val):
            errors.append(f"{key.replace('_', ' ').capitalize()} musi być w zakresie od {min_val} do {max_val}.")

    if config['arrival_lambda'] <= 0:
        errors.append("Intensywność przyjść (lambda) musi być większa od zera.")

    if not (0 <= config['class_probs'][0] <= 1 and 0 <= config['class_probs'][1] <= 1 and 0 <= config['class_probs'][2] <= 1):
        errors.append("Prawdopodobieństwa klas muszą być w zakresie od 0% do 100%.")

    if abs(sum(config['class_probs']) - 1.0) > 1e-6:
        errors.append("Suma prawdopodobieństw klas musi wynosić 100%.")

    if errors:
        raise ValueError(f"Nieprawidłowa konfiguracja: {', '.join(errors)}")

def update_config():
    try:
        CONFIG['registration_servers'] = int(registration_servers_var.get())
        CONFIG['admissions_servers'] = int(admissions_servers_var.get())
        CONFIG['gynecology_servers'] = int(gynecology_servers_var.get())
        CONFIG['predelivery_servers'] = int(predelivery_servers_var.get())
        CONFIG['delivery_servers'] = int(delivery_servers_var.get())
        CONFIG['icu_servers'] = int(icu_servers_var.get())
        CONFIG['postpartum_servers'] = int(postpartum_servers_var.get())
        CONFIG['arrival_lambda'] = float(arrival_lambda_var.get())
        
        class1_prob = float(class1_prob_var.get()) / 100
        class2_prob = float(class2_prob_var.get()) / 100
        class3_prob = float(class3_prob_var.get()) / 100

        CONFIG['class_probs'] = [class1_prob, class2_prob, class3_prob]

        validate_with_ranges(CONFIG)

        messagebox.showinfo("Sukces", "Konfiguracja została pomyślnie zaktualizowana.")
        return True
    except ValueError as e:
        messagebox.showerror("Błąd", f"Nieprawidłowe dane: {e}")
        return False

def run_simulation():
    if not update_config():
        return

    try:
        hospital = HospitalNetwork(CONFIG)

        for hour in range(100):
            new_patients = int(np.random.poisson(CONFIG['arrival_lambda']))
            for _ in range(new_patients):
                cl = np.random.choice([1, 2, 3], p=CONFIG['class_probs'])
                hospital.add_patient(hour, cl)

        hospital.simulate(total_time=100, step=0.1)
        hospital.plot_queue_lengths()

        result_bcmp = run_bcmp_analysis()
        # messagebox.showinfo("Symulacja zakończona", f"BCMP - Obciążenia węzłów: {result_bcmp['rho_i']}")
    except Exception as e:
        messagebox.showerror("Błąd", f"Wystąpił problem podczas symulacji: {e}")

root = tk.Tk()
root.title("Edycja konfiguracji i uruchamianie symulacji")

registration_servers_var = tk.StringVar(value=CONFIG['registration_servers'])
admissions_servers_var = tk.StringVar(value=CONFIG['admissions_servers'])
gynecology_servers_var = tk.StringVar(value=CONFIG['gynecology_servers'])
predelivery_servers_var = tk.StringVar(value=CONFIG['predelivery_servers'])
delivery_servers_var = tk.StringVar(value=CONFIG['delivery_servers'])
icu_servers_var = tk.StringVar(value=CONFIG['icu_servers'])
postpartum_servers_var = tk.StringVar(value=CONFIG['postpartum_servers'])
arrival_lambda_var = tk.StringVar(value=CONFIG['arrival_lambda'])
class1_prob_var = tk.StringVar(value=str(CONFIG['class_probs'][0] * 100))
class2_prob_var = tk.StringVar(value=str(CONFIG['class_probs'][1] * 100))
class3_prob_var = tk.StringVar(value=str(CONFIG['class_probs'][2] * 100))

def create_label_entry(row, label_text, variable):
    tk.Label(root, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=variable).grid(row=row, column=1, padx=5, pady=5)

create_label_entry(0, "Liczba kanałów obsługi (Rejestracja):", registration_servers_var)
create_label_entry(1, "Liczba kanałów obsługi (Izba Przyjęć):", admissions_servers_var)
create_label_entry(2, "Liczba kanałów obsługi (Oddział ginekologiczny):", gynecology_servers_var)
create_label_entry(3, "Liczba kanałów obsługi (Sala przedporodowa):", predelivery_servers_var)
create_label_entry(4, "Liczba kanałów obsługi (Sala porodowa):", delivery_servers_var)
create_label_entry(5, "Liczba kanałów obsługi (OIOM):", icu_servers_var)
create_label_entry(6, "Liczba kanałów obsługi (Sala poporodowa):", postpartum_servers_var)
create_label_entry(7, "Intensywność przybywania pacjentek [1/h]:", arrival_lambda_var)
create_label_entry(8, "Prawdopodobieństwo klasy 1 (%):", class1_prob_var)
create_label_entry(9, "Prawdopodobieństwo klasy 2 (%):", class2_prob_var)
create_label_entry(10, "Prawdopodobieństwo klasy 3 (%):", class3_prob_var)

# save_button = tk.Button(root, text="Zapisz", command=update_config)
# save_button.grid(row=11, column=0, columnspan=2, pady=10)

run_button = tk.Button(root, text="Uruchom symulację", command=run_simulation)
run_button.grid(row=12, column=0, columnspan=2, pady=10)

root.mainloop()