#Number of division on the circle
nbDiv=4
#Height of the scanner (cm)
hScan=1.5
#Number of model to process
numberOfTrees=10

for n in $( seq 1 $numberOfTrees)
do
	treeobj="t${n}.obj"
	branchobj="b${n}.obj"
	leafobj="l${n}.obj"	

	treeobj_sub="t${n}_sub.obj"
	/home/jules/Ext_Projects/trimesh2/bin.Linux64/mesh_filter $treeobj -fly ${treeobj}_1
	/home/jules/Ext_Projects/trimesh2/bin.Linux64/mesh_filter ${treeobj}_1 -fly ${treeobj_sub}
	treeobj=${treeobj_sub}

	branchobj_sub="b${n}_sub.obj"
	/home/jules/Ext_Projects/trimesh2/bin.Linux64/mesh_filter $branchobj -fly ${branchobj}_1
	/home/jules/Ext_Projects/trimesh2/bin.Linux64/mesh_filter ${branchobj}_1 -fly ${branchobj_sub}
	branchobj=${branchobj_sub}

	leafobj_sub="l${n}_sub.obj"
	/home/jules/Ext_Projects/trimesh2/bin.Linux64/mesh_filter $leafobj -fly ${leafobj}_1
	/home/jules/Ext_Projects/trimesh2/bin.Linux64/mesh_filter ${leafobj}_1 -fly ${leafobj_sub}
	leafobj=${leafobj_sub}

	treeNoExt="${treeobj%.*}"
	branchNoExt="${branchobj%.*}"
	leafNoExt="${leafobj%.*}"

	/home/jules/Software/pbrt-v2l/obj2pbrt 1.0 $treeobj
	treePbrt="$treeNoExt.pbrt"
	/home/jules/Software/pbrt-v2l/obj2pbrt 1.0 $branchobj
	branchPbrt="$branchNoExt.pbrt"
	/home/jules/Software/pbrt-v2l/obj2pbrt 1.0 $leafobj
	leafPbrt="$leafNoExt.pbrt"

	#merging both mesh
	fullobj="full${n}.obj"
	fullNoExt="${fullobj%.*}"
	/home/jules/Ext_Projects/trimesh2/bin.Linux64/mesh_cat $treeobj $branchobj $leafobj -o $fullobj
	/home/jules/Software/pbrt-v2l/obj2pbrt 1.0 $fullobj
	fullPbrt="$fullNoExt.pbrt"

	#Radius of the circle (m)
	for r in 10
	do
		for div in $( seq 1 $nbDiv )
		do
			#Compute the coordinates of the camera
			fullAng=$(echo "scale=10; 8*a(1)" | bc -l)
			angFrac=$(echo "scale=10; $fullAng / $nbDiv" | bc -l)
			ang=$(echo "scale=10; $div*$angFrac" | bc -l)
			OX=$(echo "scale=3; $r*c($ang)" | bc -l)
			OY=$(echo "scale=3; $r*s($ang)" | bc -l)
			OZ=$hScan

			treeOut="${treeNoExt}_${OX}_${OY}_${OZ}.xyz"

			#Infect the positions in the pbrt settings
			cp /home/jules/Software/pbrt-v2l/scene10000.pbrt scene.pbrt
			sed -i "s/_OX_/$OX/g" scene.pbrt
			sed -i "s/_OY_/$OY/g" scene.pbrt
			sed -i "s/_OZ_/$OZ/g" scene.pbrt
			sed -i "s/_FILEIN_/$treePbrt/g" scene.pbrt
			sed -i "s/_FILEOUT_/$treeOut/g" scene.pbrt

			#Produce the point cloud
			/home/jules/Software/pbrt-v2l/src/bin/pbrt --quiet scene.pbrt
			rm scene.pbrt

			branchOut="${branchNoExt}_${OX}_${OY}_${OZ}.xyz"

			#Infect the positions in the pbrt settings
			cp /home/jules/Software/pbrt-v2l/scene10000.pbrt scene.pbrt
			sed -i "s/_OX_/$OX/g" scene.pbrt
			sed -i "s/_OY_/$OY/g" scene.pbrt
			sed -i "s/_OZ_/$OZ/g" scene.pbrt
			sed -i "s/_FILEIN_/$branchPbrt/g" scene.pbrt
			sed -i "s/_FILEOUT_/$branchOut/g" scene.pbrt

			#Produce the point cloud
			/home/jules/Software/pbrt-v2l/src/bin/pbrt --quiet scene.pbrt
			rm scene.pbrt

			leafOut="${leafNoExt}_${OX}_${OY}_${OZ}.xyz"

			#Infect the positions in the pbrt settings
			cp /home/jules/Software/pbrt-v2l/scene10000.pbrt scene.pbrt
			sed -i "s/_OX_/$OX/g" scene.pbrt
			sed -i "s/_OY_/$OY/g" scene.pbrt
			sed -i "s/_OZ_/$OZ/g" scene.pbrt
			sed -i "s/_FILEIN_/$leafPbrt/g" scene.pbrt
			sed -i "s/_FILEOUT_/$leafOut/g" scene.pbrt

			#Produce the point cloud then voxelize it
			/home/jules/Software/pbrt-v2l/src/bin/pbrt --quiet scene.pbrt
			rm scene.pbrt

			fullOut="${fullNoExt}_${OX}_${OY}_${OZ}.xyz"

			#Infect the positions in the pbrt settings
			cp /home/jules/Software/pbrt-v2l/scene10000.pbrt scene.pbrt
			sed -i "s/_OX_/$OX/g" scene.pbrt
			sed -i "s/_OY_/$OY/g" scene.pbrt
			sed -i "s/_OZ_/$OZ/g" scene.pbrt
			sed -i "s/_FILEIN_/$fullPbrt/g" scene.pbrt
			sed -i "s/_FILEOUT_/$fullOut/g" scene.pbrt

			#Produce the point cloud
			/home/jules/Software/pbrt-v2l/src/bin/pbrt --quiet scene.pbrt
			rm scene.pbrt

			/home/jules/Project/applyLabelLeafBranchWood/applyLabelLeafBranchWood $treeOut $branchOut $leafOut $fullOut

			rm $treeOut
			rm $leafOut

		done
	done

	rm $fullPbrt
	rm $treePbrt
	rm $leafPbrt
	rm pbrt.exr
done
