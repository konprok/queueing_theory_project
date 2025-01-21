import math

def calculate_service_rate(time_in_hours):
    return 1.0 / time_in_hours

CONFIG = {
    # Intensywności obsługi:
    'registration_rate': calculate_service_rate(0.25),  # 15 min
    'registration_servers': 1,
    'registration_distribution': 'fixed',
    'registration_fixed_time': 0.25,

    'admissions_rate': calculate_service_rate(0.33),  # 20 min
    'admissions_servers': 2,

    # Ginekologia, docelowo M/M/∞
    'gynecology_rate': calculate_service_rate(36),   # 36h
    'gynecology_servers': 10,  

    # Sala przedporodowa
    'predelivery_rate': calculate_service_rate(8),  # 8h
    'predelivery_servers': 15,

    # Porodówka
    'delivery_rate': calculate_service_rate(3),      # 3h
    'delivery_servers': 6,

    # OIOM
    'icu_rate': calculate_service_rate(24),          # 24h
    'icu_servers': 1,

    # Sala poporodowa
    'postpartum_rate': calculate_service_rate(24),   # 24h
    'postpartum_servers': 25,

    # Strumień napływu pacjentek
    'arrival_lambda': 1.0,
    'class_probs': [0.65, 0.25, 0.1],  # klasa1=0.65, kl2=0.25, kl3=0.1

    #Przechodzenie między nodeami 
    
    # Rejestracja: 20% kl3 -> rejestracja, 80% -> omija
    'registration_class3_in': 0.2,

    # Izba przyjęć:
    # kl1 -> 75% sala przedporodowa, 25% porodówka
    'admissions_class1_predelivery': 0.75,
    'admissions_class1_delivery': 0.25,

    # kl2 -> 40% predelivery, 20% delivery, 40% gyn
    'admissions_class2_predelivery': 0.40,
    'admissions_class2_delivery': 0.20,
    'admissions_class2_gyn': 0.40,

    # Sala przedporodowa:
    #  kl1->100% -> porodówka
    #  kl2->50% -> kl1 +porodówka, 50%->kl2 +porodówka
    'predelivery_class2_switch_to1': 0.5,

    # Porodówka
    'delivery_class3_to2': 0.4,
    'delivery_class3_to1': 0.3,
    'delivery_class2_to1': 0.6,
    'delivery_class2_to3': 0.1,
    'delivery_class1_to3': 0.03,
    'delivery_class1_to2': 0.07,

    # OIOM
    'icu_p_newclass1': 0.5,  

    # Postpartum
    'postpartum_class2_switch_to1': 0.7
}