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
        self.generated_times = []  # Zapisz wygenerowane czasy obsługi do analizy

    def _get_service_time(self):
        if self.time_distribution == 'exponential':
            service_time = np.random.exponential(1 / self.service_rate)
        elif self.time_distribution == 'fixed':
            service_time = self.fixed_time + np.random.normal(0, 1 / self.service_rate)
        # Ustaw minimalny czas obsługi jako proporcję średniego czasu obsługi
        return max(service_time, 1 / (10 * self.service_rate))

    def add_customer(self, customer):
        if len(self.in_service) < self.servers:
            self.in_service.append(customer)
        else:
            if self.policy == 'FIFO':
                self.queue.append(customer)
            elif self.policy == 'LIFO':
                self.queue.appendleft(customer)
            elif self.policy == 'LIFO + PR':
                self.queue.append(customer)
                self.queue = deque(sorted(self.queue, key=lambda x: x['class'], reverse=True))

    def process(self, time_step):
        completed = []
        for customer in self.in_service:
            customer["remaining_time"] -= time_step
            if customer["remaining_time"] <= 0:
                completed.append(customer)
        for customer in completed:
            self.in_service.remove(customer)
        while len(self.in_service) < self.servers and self.queue:
            next_customer = self.queue.popleft()
            next_customer["remaining_time"] = self._get_service_time()
            self.in_service.append(next_customer)
        self.queue_history.append(len(self.queue))
        return completed

class HospitalNetwork:
    def __init__(self, config):
        self.registration = Queue(service_rate=config['registration_rate'], servers=config['registration_servers'],
                                   time_distribution=config['registration_distribution'], fixed_time=config['registration_fixed_time'])
        self.admissions = Queue(service_rate=config['admissions_rate'], servers=config['admissions_servers'], policy='LIFO')
        self.gynecology = Queue(service_rate=config['gynecology_rate'], servers=config['gynecology_servers'], policy='PS')
        self.delivery = Queue(service_rate=config['delivery_rate'], servers=config['delivery_servers'], policy='LIFO')
        self.icu = Queue(service_rate=config['icu_rate'], servers=config['icu_servers'], policy='PS')
        self.postpartum = Queue(service_rate=config['postpartum_rate'], servers=config['postpartum_servers'], policy='FIFO')
        self.customers = []
        self.time = 0

    def add_patient(self, arrival_time, patient_class):
        self.customers.append({"arrival_time": arrival_time, "remaining_time": 0, "class": patient_class, "location": "registration"})

    def simulate(self, total_time, time_step=1):
        while self.time < total_time:
            self.time += time_step
            new_customers = [c for c in self.customers if c["arrival_time"] == self.time]
            for customer in new_customers:
                self.registration.add_customer(customer)

            completed_registration = self.registration.process(time_step)
            for customer in completed_registration:
                customer["remaining_time"] = max(np.random.exponential(1 / self.admissions.service_rate), 1 / (10 * self.admissions.service_rate))
                customer["location"] = "admissions"
                self.admissions.add_customer(customer)

            completed_admissions = self.admissions.process(time_step)
            for customer in completed_admissions:
                # brak wejscia klasy 3 z pominieciem rejestracji
                if customer["class"] in [1, 3]:
                    customer["remaining_time"] = max(np.random.exponential(1 / self.delivery.service_rate), 1 / (10 * self.delivery.service_rate))
                    customer["location"] = "delivery"
                    self.delivery.add_customer(customer)
                elif customer["class"] == 2:
                    customer["remaining_time"] = max(np.random.exponential(1 / self.gynecology.service_rate), 1 / (10 * self.gynecology.service_rate))
                    customer["location"] = "gynecology"
                    # brakuje wyjścia do domu dla klasy 1
                    # brakuje wyjścia do innego odziały dla klasy 2
                    self.gynecology.add_customer(customer)

            completed_gynecology = self.gynecology.process(time_step)
            for customer in completed_gynecology:
                customer["remaining_time"] = max(np.random.exponential(1 / self.delivery.service_rate), 1 / (10 * self.delivery.service_rate))
                customer["location"] = "delivery"
                # dodać konwersje klasy 2 do 1 z wyjściem do dom
                self.delivery.add_customer(customer)

            completed_delivery = self.delivery.process(time_step)
            for customer in completed_delivery:
                if customer["class"] == 3:
                    customer["remaining_time"] = max(np.random.exponential(1 / self.icu.service_rate), 1 / (10 * self.icu.service_rate))
                    customer["location"] = "icu"
                    self.icu.add_customer(customer)
                elif customer["class"] in [1, 2]:
                    customer["remaining_time"] = max(np.random.exponential(1 / self.postpartum.service_rate), 1 / (10 * self.postpartum.service_rate))
                    customer["location"] = "postpartum"
                    self.postpartum.add_customer(customer)

            completed_icu = self.icu.process(time_step)
            for customer in completed_icu:
                new_class = np.random.choice([1, 2], p=[0.5, 0.5])
                customer["class"] = new_class
                if new_class == 1:
                    customer["remaining_time"] = max(np.random.exponential(1 / self.postpartum.service_rate), 1 / (10 * self.postpartum.service_rate))
                    customer["location"] = "postpartum"
                    self.postpartum.add_customer(customer)
                elif new_class == 2:
                    customer["location"] = "other_department"

            completed_postpartum = self.postpartum.process(time_step)
            for customer in completed_postpartum:
                if customer["class"] == 1:  # Klasa 1 → Dom
                    customer["location"] = "discharged"
                    print(f"Patient {customer} discharged to home.")
                elif customer["class"] == 2:  # Klasa 2 → Inny Oddział
                    customer["location"] = "other_department"
                    print(f"Patient {customer} transferred to another department.")
                # klasa 2 powinna sie dzielic pół do domu a poł na inny oddział np

        print("Symulacja zakończona.")

    def plot_queue_lengths(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.registration.queue_history, label='Rejestracja')
        plt.plot(self.admissions.queue_history, label='Izba przyjęć')
        plt.plot(self.gynecology.queue_history, label='Oddział ginekologiczny')
        plt.plot(self.delivery.queue_history, label='Porodówka')
        plt.plot(self.icu.queue_history, label='OIOM')
        plt.plot(self.postpartum.queue_history, label='Sala poporodowa')
        plt.title('Długości kolejek w czasie')
        plt.xlabel('Czas (jednostki czasu)')
        plt.ylabel('Długość kolejki')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    def calculate_service_rate(time_in_hours):
        return 1 / time_in_hours

    config = {
        'registration_rate': calculate_service_rate(0.25),
        'registration_servers': 2,
        'registration_distribution': 'fixed',
        'registration_fixed_time': 0.25,
        'admissions_rate': calculate_service_rate(0.33),
        'admissions_servers': 2,
        'gynecology_rate': calculate_service_rate(48),
        'gynecology_servers': 5,
        'delivery_rate': calculate_service_rate(3),
        'delivery_servers': 3,
        'icu_rate': calculate_service_rate(24),
        'icu_servers': 5,
        'postpartum_rate': calculate_service_rate(48),
        'postpartum_servers': 20,
    }

    hospital = HospitalNetwork(config)

    for hour in range(100):
        num_patients = np.random.poisson(1)
        for _ in range(num_patients):
            patient_class = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            hospital.add_patient(hour, patient_class)

    hospital.simulate(total_time=240, time_step=1)
    hospital.plot_queue_lengths()
