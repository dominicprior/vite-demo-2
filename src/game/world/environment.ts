// Lights.  No environment maps any more.

import { DirectionalLight, AmbientLight,
} from '../../../three/threebuild/three_module.js';
import Utils from '../utils/utils.js';

export default class Environment {
    utils: Utils;
    ambientIntensity = 0.3;
    sunlightIntensity = 0.5;
    sunlightIntensity2 = 0.1;
    ambient: AmbientLight;
    sunlight: DirectionalLight;
    sunlight2: DirectionalLight;

    constructor(utils: Utils) {
        this.utils = utils;

        this.ambient = new AmbientLight(0xffffff, this.ambientIntensity);
        this.utils.game.scene.add(this.ambient);
        this.utils.debug.gui.add(this.ambient, 'intensity', 0, 2, 0.01)
                .name('Ambient Intensity');

        this.sunlight = new DirectionalLight(0xffffff, this.sunlightIntensity);
        this.sunlight.position.set(15, 40, 7);
        this.sunlight.castShadow = true;
        this.sunlight.shadow.mapSize.width = 256;
        this.sunlight.shadow.mapSize.height = 256;
        this.sunlight.shadow.camera.top    =  10;
        this.sunlight.shadow.camera.bottom = -10;
        this.sunlight.shadow.camera.left   = -10;
        this.sunlight.shadow.camera.right  =  10;
        this.utils.game.scene.add(this.sunlight);
        this.utils.debug.gui.add(this.sunlight, 'intensity', 0, 2, 0.01)
                .name('Sunlight Intensity');

        this.sunlight2 = new DirectionalLight(0xffffff, this.sunlightIntensity2);
        this.sunlight2.position.set(-15, 40, -7);
        this.utils.game.scene.add(this.sunlight2);
        this.utils.debug.gui.add(this.sunlight2, 'intensity', 0, 2, 0.01)
                .name('Sunlight2 Intensity');
    }
}
