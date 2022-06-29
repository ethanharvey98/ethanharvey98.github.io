import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HciAtTuftsComponent } from './hci-at-tufts.component';

describe('HciAtTuftsComponent', () => {
  let component: HciAtTuftsComponent;
  let fixture: ComponentFixture<HciAtTuftsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HciAtTuftsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HciAtTuftsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
