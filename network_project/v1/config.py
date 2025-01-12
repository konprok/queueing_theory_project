import math

def calculate_service_rate(time_in_hours):
    return 1.0 / time_in_hours

CONFIG = {
    # Intensywności obsługi i liczby serwerów:
    'registration_rate': calculate_service_rate(0.25),  # 4/h
    'registration_servers': 1,
    'registration_distribution': 'fixed',
    'registration_fixed_time': 0.25,

    'admissions_rate': calculate_service_rate(0.33),  # ~3/h
    'admissions_servers': 2,

    'gynecology_rate': calculate_service_rate(36),   # ~0.02/h
    'gynecology_servers': 10,

    'delivery_rate': calculate_service_rate(3),      # ~0.33/h
    'delivery_servers': 5,

    'icu_rate': calculate_service_rate(24),          # ~0.04/h
    'icu_servers': 5,

    'postpartum_rate': calculate_service_rate(24),   # ~0.027/h
    'postpartum_servers': 25,

    # Strumień napływu pacjentów (dla symulacji)
    'arrival_lambda': 1,       # średnio 0.5 pacjentów / godz
    'class_probs': [0.65, 0.25, 0.1],  # klasa1=70%, kl2=25%, kl3=5%



    #                  Prawdopodobieństwa przejść          #

    # 1) Izba przyjęć:
    #    - klasa1 => p=0.1 do domu, 0.9 do delivery
    #    - klasa3 => 100% do delivery
    #    - klasa2 => p=0.2 out, 0.4 delivery, 0.4 gyn
    'admissions_class1_out': 0.1,
    'admissions_class3_out': 0.0,
    'admissions_class2_out': 0.2,
    'admissions_class2_delivery': 0.4, 
    'admissions_class2_gyn': 0.4,

    # 2) Ginekologia (gyn):
    #    klasa2 => p=0.1 out, 0.9 do delivery
    'gyn_class2_out': 0.1,

    # 3) Porodówka (delivery) – zmiana klasy:
    #    old=3 => 2(0.4),1(0.3),3(0.3)
    'delivery_class3_to2': 0.4,
    'delivery_class3_to1': 0.3,
    #  (reszta => 0.3 do samej3, by się sumowało do 1.0)
    #    old=2 => 1(0.6),3(0.1),2(0.3)
    'delivery_class2_to1': 0.6,
    'delivery_class2_to3': 0.1,
    #  (reszta => 0.3 => klasa2)
    #    old=1 => 3(0.03),2(0.07),1(0.90)
    'delivery_class1_to3': 0.03,
    'delivery_class1_to2': 0.07,

    # 4) OIOM (icu):
    #    new_class = np.random.choice([1,2], p=[0.5,0.5]) => 50% postpartum, 50% out
    'icu_p_newclass1': 0.5, # kl1 => postpartum
    # reszta => kl2 => out
    # (lub odwrotnie, jeśli tak w kodzie)
    
    # 5) Sala poporodowa (postpartum):
    #    if class=1 => out, if class=2 => p=0.7 => kl1 => out, else => other
    'postpartum_class2_switch_to1': 0.7
}
