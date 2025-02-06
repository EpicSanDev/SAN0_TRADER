import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import pytz

class ForexFactoryCalendar:
    def __init__(self):
        self.base_url = "https://www.forexfactory.com/calendar"
        self.calendar_data = None
        self.last_update = None

    def get_calendar(self, days_range=7):
        """Récupère le calendrier économique de ForexFactory"""
        try:
            current_time = datetime.now()
            
            # Vérifier si une mise à jour est nécessaire (toutes les heures)
            if (self.last_update is None or 
                (current_time - self.last_update).total_seconds() > 3600):

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                response = requests.get(self.base_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                events = []
                calendar_rows = soup.find_all('tr', class_='calendar_row')
                
                for row in calendar_rows:
                    try:
                        # Extraire les informations de l'événement
                        currency = row.find('td', class_='calendar__currency').text.strip()
                        impact = len(row.find_all('span', class_='impact'))  # 1-3 pour Low-High
                        title = row.find('span', class_='calendar__event-title').text.strip()
                        
                        # Extraire la date et l'heure
                        date_cell = row.find('td', class_='calendar__date')
                        time_cell = row.find('td', class_='calendar__time')
                        
                        date_str = date_cell.text.strip() if date_cell else ''
                        time_str = time_cell.text.strip() if time_cell else ''
                        
                        # Convertir en datetime
                        try:
                            event_datetime = datetime.strptime(f"{date_str} {time_str}", "%b %d %I:%M%p")
                            event_datetime = event_datetime.replace(year=current_time.year)
                        except:
                            event_datetime = current_time
                        
                        # Récupérer les valeurs
                        forecast = row.find('td', class_='calendar__forecast')
                        previous = row.find('td', class_='calendar__previous')
                        actual = row.find('td', class_='calendar__actual')
                        
                        events.append({
                            'datetime': event_datetime,
                            'currency': currency,
                            'impact': impact,
                            'event': title,
                            'forecast': forecast.text.strip() if forecast else None,
                            'previous': previous.text.strip() if previous else None,
                            'actual': actual.text.strip() if actual else None
                        })
                        
                    except Exception as e:
                        logging.error(f"Erreur lors du parsing d'un événement: {e}")
                        continue

                self.calendar_data = pd.DataFrame(events)
                self.last_update = current_time
                
            return self.calendar_data

        except Exception as e:
            logging.error(f"Erreur lors de la récupération du calendrier: {e}")
            return pd.DataFrame()

    def get_events_for_symbol(self, symbol, hours=24):
        """Récupère les événements économiques pour un symbole"""
        try:
            if not symbol or len(symbol) < 6:
                logging.error(f"Format de symbole invalide: {symbol}")
                return []

            # Extraire les devises du symbole (ex: EURUSD -> ['EUR', 'USD'])
            currencies = [symbol[:3], symbol[3:6]]
            
            # Vérifier que les devises sont valides
            for currency in currencies:
                if not currency.isalpha() or len(currency) != 3:
                    logging.error(f"Devise invalide dans le symbole {symbol}: {currency}")
                    return []
            
            # Récupérer les événements pour les deux devises
            events = []
            for currency in currencies:
                try:
                    currency_events = self.get_events_for_currency(currency, hours)
                    if currency_events:
                        for event in currency_events:
                            if isinstance(event, dict):  # Vérifier que l'événement est valide
                                events.append(event)
                            else:
                                logging.warning(f"Événement invalide ignoré pour {currency}")
                except Exception as ce:
                    logging.error(f"Erreur lors de la récupération des événements pour {currency}: {ce}")
                    continue
            
            # Trier par date/heure
            if events:
                try:
                    events.sort(key=lambda x: x.get('datetime', datetime.now()))
                except Exception as se:
                    logging.error(f"Erreur lors du tri des événements: {se}")
            
            return events
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des événements pour {symbol}: {e}")
            return []

    def get_events_for_currency(self, currency, hours=24):
        """Récupère les événements pour une devise spécifique"""
        try:
            calendar = self.get_calendar()
            if calendar.empty:
                return []
                
            current_time = datetime.now()
            time_range = current_time + timedelta(hours=hours)
            
            # Filtrer les événements
            events = calendar[
                (calendar['currency'] == currency) &
                (calendar['datetime'] <= time_range) &
                (calendar['datetime'] >= current_time)
            ]
            
            return events.to_dict('records')
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des événements par devise: {e}")
            return []
