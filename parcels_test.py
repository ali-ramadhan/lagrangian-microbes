from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile
import numpy as np
import math
from datetime import timedelta
from operator import attrgetter

fieldset = FieldSet.from_parcels("D:\\data\\moving_eddies\\moving_eddies")
pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle, lon=[3.3, 3.3], lat=[46.0, 47.8])

pset.execute(AdvectionRK4, runtime=timedelta(days=6), dt=timedelta(minutes=5),
             output_file=pset.ParticleFile(name="EddyParticlez.nc", outputdt=timedelta(hours=1)))
