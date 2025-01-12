from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class Queue:
    def __init__(self, service_rate, servers, time_distribution='exponential', 
                 fixed_time=None, policy='FIFO'):
        self.service_rate = service_rate
        self.servers = servers
        self.queue = deque()
        self.in_service = []
        self.time_distribution = time_distribution
        self.fixed_time = fixed_time
        self.queue_history = []
        self.policy = policy
        self.generated_times = []

    def _get_service_time(self):
        """Losowanie (lub ustalenie) czasu obsługi dla nowego pacjenta."""
        if self.time_distribution == 'exponential':
            service_time = np.random.exponential(1 / self.service_rate)
        elif self.time_distribution == 'fixed':
            service_time = self.fixed_time + np.random.normal(0, 1 / self.service_rate)

        return max(service_time, 1 / (10 * self.service_rate))

    def add_customer(self, customer):
        """
        Dodanie pacjenta do systemu: 
        - jeśli jest wolny serwer, trafia bezpośrednio do obsługi (in_service)
        - w przeciwnym razie do kolejki (FIFO lub LIFO + PR).
        """
        if len(self.in_service) < self.servers:
            self.in_service.append(customer)
        else:
            if self.policy == 'FIFO':
                self.queue.append(customer)
            elif self.policy == 'LIFO':
                self.queue.appendleft(customer)
            elif self.policy == 'LIFO + PR':
                self.queue.append(customer)
                # Sortowanie malejące po 'class' (wyższa klasa wyżej w kolejce)
                self.queue = deque(sorted(self.queue, key=lambda x: x['class'], reverse=True))

    def process(self, time_step):
        """
        Przetworzenie (obsługa) pacjentów w in_service przez time_step,
        Zwraca listę pacjentów, którzy w tym kroku ukończyli obsługę.
        
        """
        completed = []

        for customer in self.in_service:
            customer["remaining_time"] -= time_step

        for customer in list(self.in_service):
            if customer["remaining_time"] <= 0:
                completed.append(customer)
                self.in_service.remove(customer)

        while len(self.in_service) < self.servers and self.queue:
            next_customer = self.queue.popleft()
            next_customer["remaining_time"] = self._get_service_time()
            self.in_service.append(next_customer)

        # Zapis do historii wielkości kolejki
        self.queue_history.append(len(self.queue))
        return completed


class HospitalNetwork:
    def __init__(self, config):
        """Inicjalizacja poszczególnych oddziałów z zadanymi parametrami."""
        self.registration = Queue(
            service_rate=config['registration_rate'],
            servers=config['registration_servers'],
            time_distribution=config['registration_distribution'],
            fixed_time=config['registration_fixed_time'],
            policy='FIFO'
        )
        self.admissions = Queue(
            service_rate=config['admissions_rate'],
            servers=config['admissions_servers'],
            policy='LIFO'
        )
        self.gynecology = Queue(
            service_rate=config['gynecology_rate'],
            servers=config['gynecology_servers'],
            policy='FIFO'
        )
        self.delivery = Queue(
            service_rate=config['delivery_rate'],
            servers=config['delivery_servers'],
            policy='LIFO'
        )
        self.icu = Queue(
            service_rate=config['icu_rate'],
            servers=config['icu_servers'],
            policy='FIFO'
        )
        self.postpartum = Queue(
            service_rate=config['postpartum_rate'],
            servers=config['postpartum_servers'],
            policy='FIFO'
        )

        self.customers = []
        self.time = 0

    def add_patient(self, arrival_time, patient_class):
        """Dodaj pacjenta do listy oczekujących na przybycie do systemu."""
        self.customers.append({
            "arrival_time": arrival_time,
            "remaining_time": 0,
            "class": patient_class,
            "location": "registration"
        })

    def simulate(self, total_time, time_step=0.1):
        """
        Główna pętla symulacyjna (krokowa):
          1) Obsłuż pacjentów w każdym oddziale (process),
          2) Przenieś tych, którzy ukończyli obsługę, do kolejnych oddziałów,
          3) Dodaj nowych pacjentów, którzy przybyli w tym kroku,
          4) Zwiększ zegar o time_step.
        """
        while self.time < total_time:
            comp_reg = self.registration.process(time_step)
            comp_adm = self.admissions.process(time_step)
            comp_gyn = self.gynecology.process(time_step)
            comp_del = self.delivery.process(time_step)
            comp_icu = self.icu.process(time_step)
            comp_post = self.postpartum.process(time_step)

            # 2) Przeniesienie pacjentów:
            # 2.a) Rejestracja -> Izba przyjęć
            for customer in comp_reg:
                customer["remaining_time"] = max(
                    np.random.exponential(1 / self.admissions.service_rate),
                    1 / (10 * self.admissions.service_rate)
                )
                customer["location"] = "admissions"
                self.admissions.add_customer(customer)

            # 2.b) Izba przyjęć -> (Porodówka / Odział ginekologiczny / [OUT] Dom / [OUT] Inny odział)
            for customer in comp_adm:
                if customer["class"] == 1:
                    # 10% do domu
                    if np.random.rand() < 0.1:
                        customer["location"] = "discharged"
                    else:
                        customer["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1 / (10 * self.delivery.service_rate)
                        )
                        customer["location"] = "delivery"
                        self.delivery.add_customer(customer)

                elif customer["class"] == 3:
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.delivery.service_rate),
                        1 / (10 * self.delivery.service_rate)
                    )
                    customer["location"] = "delivery"
                    self.delivery.add_customer(customer)

                elif customer["class"] == 2:
                    # 20% -> Inny oddział, 40% -> Porodówka, 40% -> Ginekologia
                    r = np.random.rand()
                    if r < 0.2:
                        customer["location"] = "other_department"
                    elif r < 0.6:
                        customer["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1 / (10 * self.delivery.service_rate)
                        )
                        customer["location"] = "delivery"
                        self.delivery.add_customer(customer)
                    else:
                        customer["remaining_time"] = max(
                            np.random.exponential(1 / self.gynecology.service_rate),
                            1 / (10 * self.gynecology.service_rate)
                        )
                        customer["location"] = "gynecology"
                        self.gynecology.add_customer(customer)

            # 2.c) Odział ginekologiczny -> Porodówka / [OUT] Dom
            for customer in comp_gyn:
                if customer["class"] == 2:
                    # 10% do domu, 90% -> porodówka
                    if np.random.rand() < 0.1:
                        customer["location"] = "discharged"
                    else:
                        customer["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1 / (10 * self.delivery.service_rate)
                        )
                        customer["location"] = "delivery"
                        self.delivery.add_customer(customer)

            for customer in comp_del:
                old_class = customer["class"]
                r = np.random.rand()

                if old_class == 3:
                    # 40% -> klasa 2, 30% -> klasa 1, 30% -> pozostaje 3
                    if r < 0.40:
                        new_class = 2
                    elif r < 0.70:
                        new_class = 1
                    else:
                        new_class = 3

                elif old_class == 2:
                    # 60% -> klasa 1, 10% -> klasa 3, 30% -> pozostaje 2
                    if r < 0.60:
                        new_class = 1
                    elif r < 0.70:
                        new_class = 3
                    else:
                        new_class = 2

                elif old_class == 1:
                    # 3% -> klasa 3, 7% -> klasa 2, 90% -> pozostaje 1
                    if r < 0.03:
                        new_class = 3
                    elif r < 0.10:  # (0.03 + 0.07 = 0.10)
                        new_class = 2
                    else:
                        new_class = 1

                customer["class"] = new_class

                # Teraz, zależnie od nowej klasy, wysyłamy na OIOM (klasa 3) lub do Sali poporodwej (klasa 1 lub 2)
                if customer["class"] == 3:
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.icu.service_rate),
                        1 / (10 * self.icu.service_rate)
                    )
                    customer["location"] = "icu"
                    self.icu.add_customer(customer)
                else:
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.postpartum.service_rate),
                        1 / (10 * self.postpartum.service_rate)
                    )
                    customer["location"] = "postpartum"
                    self.postpartum.add_customer(customer)

            # 2.e) OIOM -> Sala poporodowa / [Out] Inny oddział
            for customer in comp_icu:
                new_class = np.random.choice([1, 2], p=[0.5, 0.5])
                customer["class"] = new_class
                if new_class == 1:
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.postpartum.service_rate),
                        1 / (10 * self.postpartum.service_rate)
                    )
                    customer["location"] = "postpartum"
                    self.postpartum.add_customer(customer)
                else:
                    customer["location"] = "other_department"

            # 2.f) Sala poporodowa -> [OUT] Dom / [OUT] Inny oddział
            for customer in comp_post:
                if customer["class"] == 1:
                    customer["location"] = "discharged"
                elif customer["class"] == 2:
                    if np.random.rand() < 0.7:
                        customer["class"] = 1
                        customer["location"] = "discharged"
                    else:
                        customer["location"] = "other_department"

            # Nowi pacjenci (którzy przybyli w tym kroku czasu)
            new_customers = [
                c for c in self.customers
                if abs(c["arrival_time"] - self.time) < 1e-9
            ]
            for customer in new_customers:
                if customer["class"] == 3:
                    # Klasa 3 -> pomija rejestrację
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.admissions.service_rate),
                        1 / (10 * self.admissions.service_rate)
                    )
                    customer["location"] = "admissions"
                    self.admissions.add_customer(customer)
                else:
                    # Klasa 1 lub 2 -> rejestracja
                    self.registration.add_customer(customer)

            # Zegar
            self.time += time_step

        print("Symulacja zakończona.")

    def plot_queue_lengths(self):
        """Wykres długości kolejek w czasie."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.registration.queue_history, label='Rejestracja')
        plt.plot(self.admissions.queue_history, label='Izba przyjęć')
        plt.plot(self.gynecology.queue_history, label='Oddział ginekologiczny')
        plt.plot(self.delivery.queue_history, label='Porodówka')
        plt.plot(self.icu.queue_history, label='OIOM')
        plt.plot(self.postpartum.queue_history, label='Sala poporodowa')
        plt.title('Długości kolejek w czasie')
        plt.xlabel('Krok symulacji')
        plt.ylabel('Długość kolejki')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    def calculate_service_rate(time_in_hours):
        return 1 / time_in_hours

    config = {
        'registration_rate': calculate_service_rate(0.25),   # 4/h (średni czas 0.25h)
        'registration_servers': 1,
        'registration_distribution': 'fixed',
        'registration_fixed_time': 0.25,

        'admissions_rate': calculate_service_rate(0.33),    # ~3/h
        'admissions_servers': 2,

        'gynecology_rate': calculate_service_rate(48),      # ~0.02/h = 48h średnio
        'gynecology_servers': 10,

        'delivery_rate': calculate_service_rate(3),         # ~0.33/h, 3h
        'delivery_servers': 2,

        'icu_rate': calculate_service_rate(24),             # ~0.04/h, 24h
        'icu_servers': 3,

        'postpartum_rate': calculate_service_rate(36),      # ~0.02/h, 48h
        'postpartum_servers': 10,
    }

    hospital = HospitalNetwork(config)

    for hour in range(100):
        num_patients = np.random.poisson(0.5)
        for _ in range(num_patients):
            patient_class = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            hospital.add_patient(hour, patient_class)

    hospital.simulate(total_time=100, time_step=0.1)

    hospital.plot_queue_lengths()