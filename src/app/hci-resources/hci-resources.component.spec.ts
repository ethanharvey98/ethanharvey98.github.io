import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HciResourcesComponent } from './hci-resources.component';

describe('HciResourcesComponent', () => {
  let component: HciResourcesComponent;
  let fixture: ComponentFixture<HciResourcesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HciResourcesComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HciResourcesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
