import numpy as np
from config import CONFIG
from simulation import HospitalNetwork
from analysis import run_bcmp_analysis

def main():
    # 1) Tworzymy obiekt hospital
    hospital = HospitalNetwork(CONFIG)

    # 2) Dodawanie pacjentów
    lam= CONFIG['arrival_lambda']  # 0.5/h
    class_probs= CONFIG['class_probs'] # [0.65,0.25,0.1]
    total_hours= 100
    for hour in range(total_hours):
        num_p = np.random.poisson(lam)
        for _ in range(num_p):
            cl= np.random.choice([1,2,3], p=class_probs)
            hospital.add_patient(hour, cl)

    # 3) Symulacja
    hospital.simulate(total_time=100, time_step=0.1)
    hospital.plot_queue_lengths()

    # 4) Analiza
    result_bcmp = run_bcmp_analysis()
    print("\nBCMP - Wyniki:")
    print("rho_i:", result_bcmp['rho_i'])
    print("K_i:", result_bcmp['K_i'])

if __name__=="__main__":
    main()