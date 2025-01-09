## Nie działa

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class Queue:
    def __init__(self, service_rate, servers, time_distribution='exponential', fixed_time=None, policy='FIFO'):
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
            # 'fixed' plus niewielkie zaburzenie normalne,
            # jeśli chcesz całkowicie stały, usuń np.random.normal(...)
            service_time = self.fixed_time + np.random.normal(0, 1 / self.service_rate)

        # Minimalny czas obsługi to 1/(10*service_rate)
        return max(service_time, 1 / (10 * self.service_rate))

    def add_customer(self, customer):
        """Dodanie pacjenta do systemu: albo bezpośrednio do obsługi (jeśli jest wolny serwer),
        albo do kolejki (zależnie od polityki)."""
        if len(self.in_service) < self.servers:
            self.in_service.append(customer)
        else:
            if self.policy == 'FIFO':
                self.queue.append(customer)
            elif self.policy == 'LIFO':
                self.queue.appendleft(customer)
            elif self.policy == 'LIFO + PR':
                self.queue.append(customer)
                # Sortowanie w dół względem 'class' - wyższa klasa wyżej w kolejce
                self.queue = deque(sorted(self.queue, key=lambda x: x['class'], reverse=True))

    def process(self, time_step):
        """Przetworzenie (obsługa) pacjentów w in_service przez czas time_step.
        Obsługa zależy od typu polityki (np. PS, FIFO, itp.).
        Zwraca listę pacjentów, którzy w tym kroku ukończyli obsługę."""
        completed = []

        # 1. Processor Sharing (PS)
        if self.policy == 'PS':
            if self.in_service:
                # Łączna "moc" serwerów to self.servers
                # Każdy pacjent dostaje time_step * (servers / liczba_pacjentów)
                time_share = (time_step * self.servers) / len(self.in_service)
                for customer in self.in_service:
                    customer["remaining_time"] -= time_share

                # Zbierz ukończonych (remaining_time <= 0)
                for customer in list(self.in_service):
                    if customer["remaining_time"] <= 0:
                        completed.append(customer)
                        self.in_service.remove(customer)

            # Gdy są wolne serwery, przesuwamy z kolejki do in_service
            while len(self.in_service) < self.servers and self.queue:
                next_customer = self.queue.popleft()
                next_customer["remaining_time"] = self._get_service_time()
                self.in_service.append(next_customer)

        # 2. Inne polityki (np. FIFO, LIFO itp.)
        else:
            # Zmniejszamy remaining_time o time_step
            for customer in self.in_service:
                customer["remaining_time"] -= time_step
            # Zbieramy pacjentów, którzy ukończyli obsługę
            for customer in list(self.in_service):
                if customer["remaining_time"] <= 0:
                    completed.append(customer)
                    self.in_service.remove(customer)

            # Jeśli są wolne serwery, przesuwamy pacjentów z kolejki
            while len(self.in_service) < self.servers and self.queue:
                next_customer = self.queue.popleft()
                next_customer["remaining_time"] = self._get_service_time()
                self.in_service.append(next_customer)

        # Zapis długości kolejki w danym kroku
        self.queue_history.append(len(self.queue))
        return completed


class HospitalNetwork:
    def __init__(self, config):
        """Inicjalizacja wszystkich oddziałów/szpitali z określonymi parametrami."""
        self.registration = Queue(
            service_rate=config['registration_rate'],
            servers=config['registration_servers'],
            time_distribution=config['registration_distribution'],
            fixed_time=config['registration_fixed_time']
        )
        self.admissions = Queue(
            service_rate=config['admissions_rate'],
            servers=config['admissions_servers'],
            policy='LIFO'
        )
        self.gynecology = Queue(
            service_rate=config['gynecology_rate'],
            servers=config['gynecology_servers'],
            policy='PS'
        )
        self.delivery = Queue(
            service_rate=config['delivery_rate'],
            servers=config['delivery_servers'],
            policy='LIFO'
        )
        self.icu = Queue(
            service_rate=config['icu_rate'],
            servers=config['icu_servers'],
            policy='PS'
        )
        self.postpartum = Queue(
            service_rate=config['postpartum_rate'],
            servers=config['postpartum_servers'],
            policy='FIFO'
        )

        self.customers = []  # Lista wszystkich pacjentów, wraz z czasem przybycia i klasą
        self.time = 0
        self.gynecology_count = 0

    def add_patient(self, arrival_time, patient_class):
        """Dodanie pacjenta do listy 'oczekujących' na wejście do systemu (przyjdzie o arrival_time)."""
        self.customers.append({
            "arrival_time": arrival_time,
            "remaining_time": 0,
            "class": patient_class,
            "location": "registration"
        })

    def simulate(self, total_time, time_step=0.1):
        """
        Główna pętla symulacyjna:
          1. Przetwarzamy (process) wszystkie kolejki (od rejestracji po salę poporodową).
             Wszyscy, którzy w tym kroku zakończą obsługę, są przenoszeni dalej (lub wypisywani).
          2. Dopiero PO obsłużeniu aktualnych pacjentów dodajemy nowych, którzy przyszli w tym kroku czasowym.
          3. Zwiększamy zegar o time_step.
          4. Powtarzamy, aż zegar > total_time.
        """
        while self.time < total_time:
            # 1) Obsługa kolejek w tej iteracji
            completed_registration = self.registration.process(time_step)
            completed_admissions = self.admissions.process(time_step)
            completed_gynecology = self.gynecology.process(time_step)
            completed_delivery = self.delivery.process(time_step)
            completed_icu = self.icu.process(time_step)
            completed_postpartum = self.postpartum.process(time_step)

            # 1.a) Przeniesienie pacjentów po rejestracji -> admissions
            for customer in completed_registration:
                customer["remaining_time"] = max(
                    np.random.exponential(1 / self.admissions.service_rate),
                    1 / (10 * self.admissions.service_rate)
                )
                customer["location"] = "admissions"
                self.admissions.add_customer(customer)

            # 1.b) Przeniesienie po admissions
            for customer in completed_admissions:
                if customer["class"] == 1:
                    # 10% szans na wyjście do domu
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
                    # 20% -> inny oddział, 80% -> ginekologia
                    if np.random.rand() < 0.2:
                        customer["location"] = "other_department"
                    else:
                        customer["remaining_time"] = max(
                            np.random.exponential(1 / self.gynecology.service_rate),
                            1 / (10 * self.gynecology.service_rate)
                        )
                        customer["location"] = "gynecology"
                        self.gynecology.add_customer(customer)
                        self.gynecology_count += 1
                        print(f"Pacjentka przeniesiona na ginekologię. Liczba: {self.gynecology_count}")

            # 1.c) Przeniesienie po ginekologii -> delivery
            for customer in completed_gynecology:
                if customer["class"] == 2:
                    # 10% -> dom, 90% -> porodówka
                    if np.random.rand() < 0.1:
                        customer["location"] = "discharged"
                    else:
                        customer["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1 / (10 * self.delivery.service_rate)
                        )
                        customer["location"] = "delivery"
                        self.delivery.add_customer(customer)

            # 1.d) Przeniesienie po porodówce -> icu lub postpartum
            for customer in completed_delivery:
                if customer["class"] == 3:
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.icu.service_rate),
                        1 / (10 * self.icu.service_rate)
                    )
                    customer["location"] = "icu"
                    self.icu.add_customer(customer)
                else:  # class 1 lub 2
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.postpartum.service_rate),
                        1 / (10 * self.postpartum.service_rate)
                    )
                    customer["location"] = "postpartum"
                    self.postpartum.add_customer(customer)

            # 1.e) Przeniesienie po icu -> postpartum lub other_department
            for customer in completed_icu:
                new_class = np.random.choice([1, 2], p=[0.5, 0.5])
                customer["class"] = new_class
                if new_class == 1:
                    customer["remaining_time"] = max(
                        np.random.exponential(1 / self.postpartum.service_rate),
                        1 / (10 * self.postpartum.service_rate)
                    )
                    customer["location"] = "postpartum"
                    self.postpartum.add_customer(customer)
                else:  # new_class == 2
                    customer["location"] = "other_department"

            # 1.f) Przeniesienie po postpartum
            for customer in completed_postpartum:
                if customer["class"] == 1:
                    # wychodzi do domu
                    customer["location"] = "discharged"
                elif customer["class"] == 2:
                    # 40% -> zmiana na class 1 i do domu, 60% -> other_department
                    if np.random.rand() < 0.4:
                        customer["class"] = 1
                        customer["location"] = "discharged"
                    else:
                        customer["location"] = "other_department"

            # 2) Dodaj nowych pacjentów dopiero po obsłudze
            #    (czyli w NASTĘPNYM kroku zaczną być obsługiwani)
            new_customers = [c for c in self.customers if abs(c["arrival_time"] - self.time) < 1e-9]
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

            # 3) Przesuń zegar symulacji
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
        'registration_servers': 5,
        'registration_distribution': 'fixed',
        'registration_fixed_time': 0.25,
        'admissions_rate': calculate_service_rate(0.33),    # ~3/h
        'admissions_servers': 5,
        'gynecology_rate': calculate_service_rate(48),      # ~0.02/h, czyli 48h średni czas
        'gynecology_servers': 1,
        'delivery_rate': calculate_service_rate(3),         # ~0.33/h, czyli 3h
        'delivery_servers': 3,
        'icu_rate': calculate_service_rate(24),             # ~0.04/h, czyli 24h
        'icu_servers': 1,
        'postpartum_rate': calculate_service_rate(48),      # ~0.02/h, czyli 48h
        'postpartum_servers': 5,
    }

    hospital = HospitalNetwork(config)

    # Generowanie pacjentów (przez 240 godzin)
    # time_step=0.1 => w sumie 2400 kroków
    for hour in range(1000):
        num_patients = np.random.poisson(3)  # średnio 1/h
        for _ in range(num_patients):
            # Tylko klasa 2 w tym przykładzie (wg Twojego p=[0,1,0])
            patient_class = np.random.choice([1, 2, 3], p=[0, 1, 0])
            hospital.add_patient(arrival_time=hour, patient_class=patient_class)

    # Symulacja przez 240 godzin (z krokiem 0.1h)
    hospital.simulate(total_time=1000, time_step=0.1)

    # Wykres długości kolejek
    hospital.plot_queue_lengths()
