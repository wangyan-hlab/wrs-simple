import copy
import math
import modeling.geometric_model as gm
import modeling.dynamics.bullet.bdbody as bdb


class DynamicModel(gm.GeometricModel):
    """
    load an object as a bullet dynamics model
    author: weiwei
    date: 20190627
    """

    def __init__(self, initor, mass=None, betransparency=True, cm_cdtype="box", cm_expradius=None,
                 restitution=0, allowdeactivation=False, allowccd=True, friction=.2, dynamic=False,
                 dyn_cdtype="convex", name="bdm"):
        """
        :param initor:
        :param mass:
        :param betransparency:
        :param cm_cdtype:
        :param cm_expradius:
        :param restitution:
        :param allowdeactivation:
        :param allowccd:
        :param friction:
        :param dynamic:
        :param dyn_cdtype: "convex", "triangle", etc.
        :param name:
        """
        # if isinstance(initor, BDModel):
        #     super().__init__(initor.objcm, )
        #     self.__objcm = copy.deepcopy(initor.objcm)
        #     self.__objbdb = initor.objbdb.copy()
        #     base.physicsworld.attach(self.__objbdb)
        # else:
        super().__init__(initor.objcm, btransparency=betransparency, type=cm_cdtype, cm_expradius=None,
                         name="defaultname")
        if mass is None:
            mass = 0
        self._bdb = bdb.BDBody(self.objcm, type=dyn_cdtype, mass=mass, restitution=restitution,
                               allow_deactivation=allowdeactivation, allow_ccd=allowccd, friction=friction, name="bdm")
        base.physicsworld.attach(self.__objbdb)

    @property
    def bdb(self):
        # read-only property
        return self._bdb

    def setpos(self, npvec3):
        """
        overwrite the parent's setpos to additional manipuate bdb
        :param npvec3
        :return:
        """
        homomat_bdb = self._bdb.get_homomat()
        homomat_bdb[:3, 3] = npvec3
        self._bdb.set_homomat(homomat_bdb)
        super().sethomomat(homomat_bdb)

    def getpos(self):
        homomat_bdb = self._bdb.pos()
        self._bdb.set_homomat(homomat_bdb)
        super().sethomomat(homomat_bdb)

        return self.__objcm.objnp.getPos()

    def setMat(self, pandamat4):
        self.__objbdb.set_homomat(base.pg.mat4ToNp(pandamat4))
        self.__objcm.objnp.setMat(pandamat4)
        # self.__objcm.objnp.setMat(base.pg.np4ToMat4(self.objbdb.gethomomat()))

    def sethomomat(self, npmat4):
        self.__objbdb.set_homomat(npmat4)
        self.__objcm.set_homomat(npmat4)

    def setRPY(self, roll, pitch, yaw):
        """
        set the pose of the object using rpy

        :param roll: degree
        :param pitch: degree
        :param yaw: degree
        :return:

        author: weiwei
        date: 20190513
        """

        currentmat = self.__objbdb.get_homomat()
        currentmatnp = base.pg.mat4ToNp(currentmat)
        newmatnp = rm.rotmat_from_euler(roll, pitch, yaw, axes="sxyz")
        self.setMat(base.pg.npToMat4(newmatnp, currentmatnp[:, 3]))

    def getRPY(self):
        """
        get the pose of the object using rpy

        :return: [r, p, y] in degree

        author: weiwei
        date: 20190513
        """

        currentmat = self.objcm.getMat()
        currentmatnp = base.pg.mat4ToNp(currentmat)
        rpy = rm.euler_from_matrix(currentmatnp[:3, :3], axes="sxyz")
        return np.array([rpy[0], rpy[1], rpy[2]])

    def getMat(self, rel=None):
        return self.objcm.getMat(rel)

    def gethomomat(self, rel=None):
        pandamat4 = self.getMat(rel)
        return base.pg.mat4ToNp(pandamat4)

    def setMass(self, mass):
        self.__objbdb.setMass(mass)

    def reparentTo(self, objnp):
        """
        objnp must be base.render

        :param objnp:
        :return:

        author: weiwei
        date: 20190627
        """

        # if isinstance(objnp, cm.CollisionModel):
        #     self.__objcm.objnp.reparentTo(objnp.objnp)
        # elif isinstance(objnp, NodePath):
        #     self.__objcm.objnp.reparentTo(objnp)
        # else:
        #     print("NodePath.reparent_to() argument 1 must be environment.CollisionModel or panda3d.core.NodePath")
        if objnp is not base.render:
            print("This bullet dynamics model doesnt support to plot to non base.render nodes!")
            raise ValueError("Value Error!")
        else:
            self.__objcm.objnp.reparentTo(objnp)
        # self.setMat(self.__objcm.getMat())
        # print(self.objbdb.gethomomat())
        self.__objcm.objnp.setMat(base.pg.np4ToMat4(self.objbdb.get_homomat()))

    def removeNode(self):
        self.__objcm.objnp.removeNode()
        base.physicsworld.remove(self.__objbdb)

    def detachNode(self):
        self.__objcm.objnp.detachNode()

    def showcn(self):
        # reattach to bypass the failure of deepcopy
        self.__cdnp.removeNode()
        self.__cdnp = self.__objnp.attachNewNode(self.__cdcn)
        self.__cdnp.show()

    def showLocalFrame(self):
        self.__localframe = base.pggen.genAxis()
        self.__localframe.reparentTo(self.objnp)

    def unshowLocalFrame(self):
        if self.__localframe is not None:
            self.__localframe.removeNode()
            self.__localframe = None

    def unshowcn(self):
        self.__cdnp.hide()

    def copy(self):
        return BDModel(self)


if __name__ == "__main__":
    import os
    import numpy as np
    import basis.robot_math as rm
    import pandaplotutils.pandactrl as pc
    import random
    import basis

    base = pc.World(camp=[1000, 300, 1000], lookatpos=[0, 0, 0], toggledebug=False)
    base.setFrameRateMeter(True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'block.stl')
    bunnycm = BDModel(objpath, mass=1, shapetype="convex")

    objpath = os.path.join(basis.__path__[0], 'objects', 'bowlblock.stl')
    bunnycm2 = BDModel(objpath2, mass=0, shapetype="triangle", dynamic=False)
    bunnycm2.setColor(0, 0.7, 0.7, 1.0)
    # bunnycm2.reparentTo(base.render)
    bunnycm2.setPos(0, 0, 0)
    base.attachRUD(bunnycm2)


    def update(bunnycm, task):
        if base.inputmgr.keyMap['space'] is True:
            for i in range(300):
                bunnycm1 = bunnycm.copy()
                bunnycm1.setMass(.1)
                # bunnycm1.setColor(0.7, 0, 0.7, 1.0)
                bunnycm1.setColor(random.random(), random.random(), random.random(), 1.0)
                # bunnycm1.reparentTo(base.render)
                # rotmat = rm.rodrigues([0,0,1], 15)
                rotmat = rm.rotmat_from_euler(0, 0, 15)
                z = math.floor(i / 100)
                y = math.floor((i - z * 100) / 10)
                x = i - z * 100 - y * 10
                print(x, y, z, "\n")
                bunnycm1.setMat(base.pg.npToMat4(rotmat, np.array([x * 15 - 70, y * 15 - 70, 150 + z * 15])))
                base.attachRUD(bunnycm1)
        base.inputmgr.keyMap['space'] = False
        return task.cont


    base.pggen.plotAxis(base.render)
    taskMgr.add(update, "addobject", extraArgs=[bunnycm], appendTask=True)

    base.run()
