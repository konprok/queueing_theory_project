from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from config import CONFIG

class Queue:
    def __init__(self, service_rate, servers, time_distribution='exponential', fixed_time=None, policy='FIFO'):
        if servers <= 0:
            raise ValueError("Liczba serwerów musi być większa od zera.")
        if service_rate <= 0:
            raise ValueError("Stawka obsługi musi być większa od zera.")
        self.service_rate = service_rate
        self.servers = servers
        self.queue = deque()
        self.in_service = []
        self.time_distribution = time_distribution
        self.fixed_time = fixed_time
        self.queue_history = []
        self.policy = policy

    def _get_service_time(self):
        if self.time_distribution == 'exponential':
            service_time = np.random.exponential(1 / self.service_rate)
        else:
            service_time = self.fixed_time + np.random.normal(0, 1 / self.service_rate)
        return max(service_time, 1/(10*self.service_rate))

    def add_customer(self, customer):
        # M/M/∞ => wrzucamy do in_service bez limitu
        if self.policy == 'IS':
            self.in_service.append(customer)
            return

        # Dla innych kolejek (ograniczona liczba serwerów)
        if len(self.in_service) < self.servers:
            self.in_service.append(customer)
        else:
            if self.policy == 'FIFO':
                self.queue.append(customer)
            elif self.policy == 'LIFO':
                self.queue.appendleft(customer)
            elif self.policy == 'FIFO + PR':
                # "PR" -> sort malejąco po "class"
                self.queue.append(customer)
                self.queue = deque(
                    sorted(self.queue, key=lambda x: x['class'], reverse=True)
                )

    def process(self, time_step):
        completed = []
        # Zmniejszamy remaining_time
        for c in self.in_service:
            c["remaining_time"] -= time_step

        # Usuwamy klientów, którzy skończyli
        for c in list(self.in_service):
            if c["remaining_time"] <= 0:
                completed.append(c)
                self.in_service.remove(c)

        # Jeśli policy!='IS', to możemy przenosić z kolejki
        if self.policy != 'IS':
            while len(self.in_service) < self.servers and self.queue:
                nxt = self.queue.popleft()
                nxt["remaining_time"] = self._get_service_time()
                self.in_service.append(nxt)

        self.queue_history.append(len(self.queue))
        return completed


class HospitalNetwork:
    def __init__(self, config):
        self.cfg = config
        # Węzły:
        self.registration = Queue(
            config['registration_rate'], config['registration_servers'],
            time_distribution=config['registration_distribution'],
            fixed_time=config['registration_fixed_time'],
            policy='FIFO'
        )
        self.admissions = Queue(
            config['admissions_rate'], config['admissions_servers'],
            policy='FIFO + PR'
        )
        # Ginekologia => M/M/∞
        self.gynecology = Queue(
            config['gynecology_rate'], config['gynecology_servers'],
            policy='IS'
        )
        # Sala przedporodowa => FIFO
        self.predelivery = Queue(
            config['predelivery_rate'], config['predelivery_servers'],
            policy='FIFO'
        )
        # Porodówka => FIFO+PR
        self.delivery = Queue(
            config['delivery_rate'], config['delivery_servers'],
            policy='FIFO + PR'
        )
        # OIOM => IS (M/M/∞ w symulacji)
        self.icu = Queue(
            config['icu_rate'], config['icu_servers'],
            policy='IS'
        )
        # Sala poporodowa => FIFO (M/M/c w BCMP)
        self.postpartum = Queue(
            config['postpartum_rate'], config['postpartum_servers'],
            policy='FIFO'
        )

        self.time = 0.0
        self.customers = []

    def add_patient(self, arrival_time, patient_class):
        self.customers.append({
            "arrival_time": arrival_time,
            "remaining_time": 0.0,
            "class": patient_class,  # klasa: 1=bez kompl.,2=zkompl.,3=krytyczne
            "location": None
        })

    def simulate(self, total_time, step=0.1):
        while self.time < total_time:
            reg_done = self.registration.process(step)
            adm_done = self.admissions.process(step)
            gyn_done = self.gynecology.process(step)
            pre_done = self.predelivery.process(step)
            del_done = self.delivery.process(step)
            icu_done = self.icu.process(step)
            post_done= self.postpartum.process(step)

            # Rejestracja -> Izba
            for c in reg_done:
                c["remaining_time"] = max(
                    np.random.exponential(1 / self.admissions.service_rate),
                    1/(10*self.admissions.service_rate)
                )
                c["location"] = "admissions"
                self.admissions.add_customer(c)

            # Izba
            for c in adm_done:
                if c["class"] == 1:
                    # kl1 => 75% predelivery, 25% delivery
                    p_pre = self.cfg['admissions_class1_predelivery']
                    if np.random.rand() < p_pre:
                        c["remaining_time"] = max(
                            np.random.exponential(1/self.predelivery.service_rate),
                            1/(10*self.predelivery.service_rate)
                        )
                        c["location"] = "predelivery"
                        self.predelivery.add_customer(c)
                    else:
                        c["remaining_time"] = max(
                            np.random.exponential(1/self.delivery.service_rate),
                            1/(10*self.delivery.service_rate)
                        )
                        c["location"] = "delivery"
                        self.delivery.add_customer(c)

                elif c["class"] == 2:
                    # kl2 => 40% predelivery, 20% delivery, 40% gyn
                    p_pre = self.cfg['admissions_class2_predelivery']
                    p_del = self.cfg['admissions_class2_delivery']
                    p_gyn = self.cfg['admissions_class2_gyn']
                    x = np.random.rand()
                    if x < p_pre:
                        c["remaining_time"] = max(
                            np.random.exponential(1/self.predelivery.service_rate),
                            1/(10*self.predelivery.service_rate)
                        )
                        c["location"] = "predelivery"
                        self.predelivery.add_customer(c)
                    elif x < p_pre + p_del:
                        c["remaining_time"] = max(
                            np.random.exponential(1/self.delivery.service_rate),
                            1/(10*self.delivery.service_rate)
                        )
                        c["location"] = "delivery"
                        self.delivery.add_customer(c)
                    else:
                        # do ginekologii
                        c["remaining_time"] = max(
                            np.random.exponential(1/self.gynecology.service_rate),
                            1/(10*self.gynecology.service_rate)
                        )
                        c["location"] = "gynecology"
                        self.gynecology.add_customer(c)

                elif c["class"] == 3:
                    # kl3 => 100% do delivery
                    c["remaining_time"] = max(
                        np.random.exponential(1/self.delivery.service_rate),
                        1/(10*self.delivery.service_rate)
                    )
                    c["location"] = "delivery"
                    self.delivery.add_customer(c)
            
            # Ginekologia
            for c in gyn_done:
                # 10% out, 90% -> delivery
                if np.random.rand() < 0.1:
                    c["location"] = "discharged"
                else:
                    c["remaining_time"] = max(
                        np.random.exponential(1 / self.delivery.service_rate),
                        1 / (10 * self.delivery.service_rate)
                    )
                    c["location"] = "delivery"
                    self.delivery.add_customer(c)


            # Sala przedporodowa
            for c in pre_done:
                if c["class"] == 1:
                    # kl1 => 100% do porodówki
                    c["remaining_time"] = max(
                        np.random.exponential(1/self.delivery.service_rate),
                        1/(10*self.delivery.service_rate)
                    )
                    c["location"] = "delivery"
                    self.delivery.add_customer(c)
                elif c["class"] == 2:
                    # 50% => kl1, 50% => kl2
                    if np.random.rand() < self.cfg['predelivery_class2_switch_to1']:
                        c["class"] = 1
                    c["remaining_time"] = max(
                        np.random.exponential(1/self.delivery.service_rate),
                        1/(10*self.delivery.service_rate)
                    )
                    c["location"] = "delivery"
                    self.delivery.add_customer(c)

            # Porodówka
            for c in del_done:
                old = c["class"]
                r = np.random.rand()
                if old == 3:
                    p_3to2 = self.cfg['delivery_class3_to2']
                    p_3to1 = self.cfg['delivery_class3_to1']
                    if r < p_3to2:
                        new = 2
                    elif r < p_3to2 + p_3to1:
                        new = 1
                    else:
                        new = 3
                elif old == 2:
                    p_2to1 = self.cfg['delivery_class2_to1']
                    p_2to3 = self.cfg['delivery_class2_to3']
                    if r < p_2to1:
                        new = 1
                    elif r < p_2to1 + p_2to3:
                        new = 3
                    else:
                        new = 2
                elif old == 1:
                    p_1to3 = self.cfg['delivery_class1_to3']
                    p_1to2 = self.cfg['delivery_class1_to2']
                    if r < p_1to3:
                        new = 3
                    elif r < p_1to3 + p_1to2:
                        new = 2
                    else:
                        new = 1
                c["class"] = new

                if new == 3:
                    # do OIOM
                    c["remaining_time"] = max(
                        np.random.exponential(1/self.icu.service_rate),
                        1/(10*self.icu.service_rate)
                    )
                    c["location"] = "icu"
                    self.icu.add_customer(c)
                else:
                    # kl1/kl2 => postpartum
                    c["remaining_time"] = max(
                        np.random.exponential(1/self.postpartum.service_rate),
                        1/(10*self.postpartum.service_rate)
                    )
                    c["location"] = "postpartum"
                    self.postpartum.add_customer(c)

            # OIOM
            for c in icu_done:
                if np.random.rand() < self.cfg['icu_p_newclass1']:
                    c["class"] = 1
                    c["remaining_time"] = max(
                        np.random.exponential(1/self.postpartum.service_rate),
                        1/(10*self.postpartum.service_rate)
                    )
                    c["location"] = "postpartum"
                    self.postpartum.add_customer(c)
                else:
                    c["class"] = 2
                    c["location"] = "other_department"

            # Sala poporodowa
            for c in post_done:
                if c["class"] == 1:
                    c["location"] = "discharged"
                elif c["class"] == 2:
                    # 70% => kl1 => out, 30% => other
                    if np.random.rand() < self.cfg['postpartum_class2_switch_to1']:
                        c["class"] = 1
                        c["location"] = "discharged"
                    else:
                        c["location"] = "other_department"

            # Nowi pacjenci
            new_arrivals = [x for x in self.customers
                            if abs(x["arrival_time"] - self.time) < 1e-9]
            for c in new_arrivals:
                if c["class"] == 3:
                    # 20% -> rejestracja, 80% -> izba
                    if np.random.rand() < self.cfg['registration_class3_in']:
                        c["location"] = "registration"
                        self.registration.add_customer(c)
                    else:
                        c["remaining_time"] = max(
                            np.random.exponential(1/self.admissions.service_rate),
                            1/(10*self.admissions.service_rate)
                        )
                        c["location"] = "admissions"
                        self.admissions.add_customer(c)
                else:
                    # kl1,kl2 => rejestracja
                    c["location"] = "registration"
                    self.registration.add_customer(c)

            self.time += step

        print("Symulacja zakończona.")

    def plot_queue_lengths(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.registration.queue_history, label='Rejestracja')
        plt.plot(self.admissions.queue_history, label='Izba przyjęć')
        plt.plot(self.gynecology.queue_history, label='O.ginekologiczny')
        plt.plot(self.predelivery.queue_history, label='S.przedporodowa')
        plt.plot(self.delivery.queue_history, label='S.porodowa')
        plt.plot(self.icu.queue_history, label='OIOM')
        plt.plot(self.postpartum.queue_history, label='S.poporodowa')
        plt.title('Długości kolejek w czasie')
        plt.xlabel('Krok symulacji')
        plt.ylabel('Liczba oczekujących')
        plt.legend()
        plt.grid(True)
        plt.show()