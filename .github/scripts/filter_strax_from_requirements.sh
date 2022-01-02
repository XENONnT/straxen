pwd
grep -v 'strax' requirements.txt &> temp_requirements.txt
rm requirements.txt
mv temp_requirements.txt requirements.txt
cat requirements.txt
