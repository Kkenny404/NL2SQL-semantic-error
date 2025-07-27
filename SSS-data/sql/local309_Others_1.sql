select races.year,
       drivers.forename || ' ' || drivers.surname as driver,
       constructors.name as constructor
from results
left join races on results.race_id = races.race_id
left join drivers on results.driver_id = drivers.driver_id
left join constructors on results.constructor_id = constructors.constructor_id
order by races.year;