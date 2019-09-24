#!/bin/bash/

# sources
source activate phd

# Get the shoreline for varios locations at once...

# Nobbys Beach - 2017
echo ""
echo "-> Detecting over-running for Nobbys Beach"
N=011
for i in {000..011}; do
  echo "--> processing timestack "$i" of "$N
  python detect_overrunning.py '../hyper/overrunning/20171102-'$i'-NB.json' > /dev/null
done
exit

# One Mile Beach
echo ""
echo "-> Detecting over-running for One Mile Beach"
N=011
for i in {000..011}; do
  echo "--> processing timestack "$i" of "$N
  python detect_overrunning.py -i '../hyper/overrunning/20140807-'$i'-OMB.json' > /dev/null
done

# Werri Beach
echo ""
echo "-> Detecting over-running for Werri Beach"
N=011
for i in {000..011}; do
  echo "--> processing timestack "$i" of "$N
  python detect_overrunning.py -i '../hyper/overrunning/20140816-'$i'-WB.json' > /dev/null
done

# Moreton Inland
echo ""
echo "-> Detecting over-running for Moreton Island"
N=011
for i in {000..011}; do
  echo "--> processing timestack "$i" of "$N
  python detect_overrunning.py '../hyper/overrunning/20161220-'$i'-MI.json' > /dev/null
done

# Frazer Beach
echo ""
echo "-> Detecting over-running for Frazer Beach"
N=011
for i in {000..011}; do
  echo "--> processing timestack "$i" of "$N
  python detect_overrunning.py '../hyper/overrunning/20180424-'$i'-FB.json' > /dev/null
done


# Seven Mile Beach - 2018
echo ""
echo "-> Detecting over-running for Seven Mile Beach"
N=011
for i in {000..011}; do
  echo "--> processing timestack "$i" of "$N
  python detect_overrunning.py '../hyper/overrunning/20180614-'$i'-SMB.json' > /dev/null
done
